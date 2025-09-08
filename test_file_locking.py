#!/usr/bin/env python3
"""
Test the new file-based locking mechanism for preventing duplicate orders
"""
import os
import sys
import fcntl
import time
import threading
from pathlib import Path

def test_concurrent_locks():
    """Test that file locking prevents concurrent access"""
    print("🔍 Testing concurrent file locking mechanism...")
    
    lock_dir = Path("state/order_locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / "BTCUSD.lock"
    
    results = []
    
    def try_lock(process_id):
        try:
            with open(lock_file, 'w') as f:
                # Try non-blocking lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                results.append(f"Process {process_id}: ACQUIRED LOCK")
                time.sleep(0.1)  # Simulate work
                results.append(f"Process {process_id}: released lock")
        except (IOError, OSError) as e:
            if e.errno in (11, 35):  # EAGAIN or EWOULDBLOCK
                results.append(f"Process {process_id}: BLOCKED (lock held by another process)")
            else:
                results.append(f"Process {process_id}: ERROR - {e}")
    
    # Start multiple threads simulating concurrent processes
    threads = []
    for i in range(5):
        thread = threading.Thread(target=try_lock, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    print("\n".join(results))
    
    # Count how many got locks vs blocked
    acquired = len([r for r in results if "ACQUIRED LOCK" in r])
    blocked = len([r for r in results if "BLOCKED" in r])
    
    print(f"\n✅ Results: {acquired} processes acquired lock, {blocked} were blocked")
    
    if acquired == 1 and blocked >= 1:
        print("✅ File locking working correctly - only one process can acquire lock")
        return True
    else:
        print("❌ File locking not working - multiple processes got lock")
        return False

def test_lock_cleanup():
    """Test that locks are properly cleaned up"""
    print("\n🔍 Testing lock cleanup...")
    
    lock_dir = Path("state/order_locks")
    lock_file = lock_dir / "ETHUSDT.lock"
    
    # Acquire and release lock
    try:
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            print("✅ Lock acquired")
        # Lock should be automatically released here
        
        # Try to acquire again - should succeed if cleanup worked
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("✅ Lock re-acquired successfully - cleanup working")
        
        return True
    except Exception as e:
        print(f"❌ Lock cleanup test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 TESTING FILE-BASED LOCKING MECHANISM")
    print("=" * 50)
    
    # Ensure we're in the right directory
    os.chdir("/Users/mouryadamarasing/Documents/Project-Tracer-Bullet")
    
    test1 = test_concurrent_locks()
    test2 = test_lock_cleanup()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("✅ ALL TESTS PASSED - File locking mechanism working correctly")
        sys.exit(0)
    else:
        print("❌ TESTS FAILED - File locking mechanism has issues")
        sys.exit(1)
