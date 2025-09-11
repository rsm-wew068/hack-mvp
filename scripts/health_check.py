#!/usr/bin/env python3
"""Health check script for the Music AI Recommendation System."""

import asyncio
import aiohttp
import time
import sys
from typing import Dict, Any


class HealthChecker:
    """Health checker for all services."""
    
    def __init__(self):
        self.services = {
            'vllm': 'http://localhost:8002/health',
            'backend': 'http://localhost:8000/health',
            'frontend': 'http://localhost:8501/_stcore/health',
            'flower': 'http://localhost:5555'
        }
    
    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all services."""
        results = {}
        
        for service, url in self.services.items():
            try:
                if service == 'redis':
                    results[service] = await self.check_redis()
                else:
                    results[service] = await self.check_http_service(url)
            except Exception as e:
                results[service] = {'status': 'unhealthy', 'error': str(e)}
        
        return results
    
    async def check_http_service(self, url: str) -> Dict[str, Any]:
        """Check HTTP service health."""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        return {
                            'status': 'healthy',
                            'response_time': f"{response_time:.3f}s",
                            'status_code': response.status
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'status_code': response.status,
                            'response_time': f"{response_time:.3f}s"
                        }
        except asyncio.TimeoutError:
            return {
                'status': 'timeout',
                'response_time': '>10s'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            import redis.asyncio as redis
            
            r = redis.Redis.from_url('redis://localhost:6379')
            await r.ping()
            return {'status': 'healthy'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}


async def main():
    """Main health check function."""
    print("🏥 Running health checks...")
    
    checker = HealthChecker()
    results = await checker.check_all_services()
    
    print("\n📊 Health Check Results:")
    print("=" * 50)
    
    all_healthy = True
    for service, result in results.items():
        status = result['status']
        emoji = "✅" if status == 'healthy' else "❌"
        
        print(f"{emoji} {service.upper()}: {status}")
        
        if 'response_time' in result:
            print(f"   Response time: {result['response_time']}")
        
        if 'error' in result:
            print(f"   Error: {result['error']}")
            all_healthy = False
        
        print()
    
    if all_healthy:
        print("🎉 All services are healthy!")
        return 0
    else:
        print("⚠️  Some services are unhealthy. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
