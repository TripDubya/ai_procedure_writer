import prometheus_client
from prometheus_client import Counter, Histogram
import psutil
import logging

logger = logging.getLogger(__name__)

# Metrics
REQUESTS = Counter('procedure_requests_total', 'Total procedure generation requests')
GENERATION_TIME = Histogram('procedure_generation_seconds', 'Time spent generating procedures')
MODEL_LOAD_TIME = Histogram('model_load_seconds', 'Time spent loading the model')
SYSTEM_MEMORY = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
GPU_MEMORY = Gauge('gpu_memory_usage_bytes', 'GPU memory usage in bytes')

class SystemMonitor:
    @staticmethod
    def update_metrics():
        try:
            # Update system metrics
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY.set(memory.used)

            # Update GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                GPU_MEMORY.set(gpu_memory)

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    @staticmethod
    def start_prometheus_server(port=9090):
        try:
            prometheus_client.start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {str(e)}")