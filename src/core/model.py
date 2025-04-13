import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from typing import Optional, Tuple, Dict
import os
import json
from datetime import datetime
import hashlib
from pathlib import Path
import logging
from functools import wraps
import tempfile

logger = logging.getLogger(__name__)

def requires_auth(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_authenticated():
            raise PermissionError("Authentication required for this operation")
        return func(self, *args, **kwargs)
    return wrapper

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model: Optional[GPT2LMHeadModel] = None
        self.tokenizer: Optional[GPT2Tokenizer] = None
        self.model_config: Dict = self._load_config()
        self.current_model_hash: Optional[str] = None
        
    def _load_config(self) -> Dict:
        config_path = self.model_dir / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "version": "0.1.0",
            "last_updated": None,
            "checkpoints": []
        }

    def _save_config(self):
        with open(self.model_dir / "model_config.json", 'w') as f:
            json.dump(self.model_config, f, indent=2)

    def _calculate_model_hash(self) -> str:
        """Calculate a hash of the model's state dict for versioning"""
        state_dict = self.model.state_dict()
        model_bytes = pickle.dumps(state_dict)
        return hashlib.sha256(model_bytes).hexdigest()

    @requires_auth
    def initialize_model(self, model_name: str = 'gpt2') -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
        """Initialize the model with security checks and versioning"""
        try:
            # Load model and tokenizer with security verification
            if not self._verify_model_source(model_name):
                raise SecurityError("Model source verification failed")

            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')

            # Calculate and store model hash
            self.current_model_hash = self._calculate_model_hash()
            
            # Update config
            self._update_model_metadata()
            
            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _verify_model_source(self, model_name: str) -> bool:
        """Verify the model source is trusted"""
        trusted_sources = ['gpt2', 'gpt2-medium', 'gpt2-large']
        return model_name in trusted_sources

    @requires_auth
    def save_checkpoint(self, checkpoint_name: str):
        """Save a model checkpoint with versioning"""
        try:
            checkpoint_path = self.model_dir / f"checkpoint_{checkpoint_name}"
            
            # Save model and tokenizer
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)

            # Update metadata
            checkpoint_info = {
                "name": checkpoint_name,
                "date": datetime.now().isoformat(),
                "hash": self.current_model_hash,
                "path": str(checkpoint_path)
            }
            self.model_config["checkpoints"].append(checkpoint_info)
            self._save_config()

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    @requires_auth
    def load_checkpoint(self, checkpoint_name: str):
        """Load a specific model checkpoint"""
        try:
            checkpoint = next(
                (cp for cp in self.model_config["checkpoints"] 
                 if cp["name"] == checkpoint_name), None
            )
            
            if not checkpoint:
                raise ValueError(f"Checkpoint {checkpoint_name} not found")

            checkpoint_path = Path(checkpoint["path"])
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint files not found at {checkpoint_path}")

            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)

            if torch.cuda.is_available():
                self.model = self.model.to('cuda')

            self.current_model_hash = checkpoint["hash"]

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise

    def _update_model_metadata(self):
        """Update model metadata after changes"""
        self.model_config["last_updated"] = datetime.now().isoformat()
        self._save_config()

    @requires_auth
    def verify_model_integrity(self) -> bool:
        """Verify the current model hasn't been tampered with"""
        if not self.current_model_hash:
            return False
        return self._calculate_model_hash() == self.current_model_hash

    @requires_auth
    def train_model(self, training_data: str, epochs: int = 1) -> bool:
        """Train the model on new data"""
        try:
            # Create temporary file for training data
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
                f.write(training_data)
                temp_file = f.name

            # Prepare dataset
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=temp_file,
                block_size=128
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir="./training_output",
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=4,
                save_steps=10000,
                save_total_limit=2,
                logging_steps=100,
                logging_dir="./logs"
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )

            # Train the model
            trainer.train()
            
            # Update model hash after training
            self.current_model_hash = self._calculate_model_hash()
            
            # Cleanup
            os.unlink(temp_file)
            
            return True
        
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
