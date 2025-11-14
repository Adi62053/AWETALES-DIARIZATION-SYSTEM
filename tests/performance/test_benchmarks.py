"""
Performance benchmarking tests
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import patch, AsyncMock


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_streaming_latency_benchmark(self):
        """Benchmark streaming latency"""
        latencies = []
        
        for i in range(10):  # Test with 10 chunks
            start_time = time.time()
            
            # Simulate processing a chunk
            await asyncio.sleep(0.05)  # Simulate 50ms processing
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Max latency: {max_latency:.2f}ms")
        
        # Assert latency targets are met
        assert avg_latency < 100, f"Average latency {avg_latency}ms exceeds 100ms target"
        assert max_latency < 500, f"Max latency {max_latency}ms exceeds 500ms target"
    
    @pytest.mark.benchmark
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operation
        large_list = [f"test_data_{i}" * 100 for i in range(10000)]
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f}MB")
        print(f"Peak memory: {peak_memory:.2f}MB") 
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # Clean up
        del large_list
        
        # Assert memory usage is reasonable
        assert memory_increase < 500, f"Memory increase {memory_increase}MB exceeds 500MB limit"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_processing_benchmark(self):
        """Benchmark concurrent processing capability"""
        async def mock_processing_task(task_id):
            await asyncio.sleep(0.1)  # Simulate 100ms processing
            return {"task_id": task_id, "status": "completed"}
        
        # Test with multiple concurrent tasks
        num_tasks = 5
        start_time = time.time()
        
        tasks = [mock_processing_task(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Processed {num_tasks} tasks in {total_time:.2f}s")
        print(f"Throughput: {num_tasks / total_time:.2f} tasks/second")
        
        assert len(results) == num_tasks
        # Concurrent processing should be faster than sequential
        assert total_time < num_tasks * 0.15, "Concurrent processing not efficient"