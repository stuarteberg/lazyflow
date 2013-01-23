# Built-in
import atexit
import collections
import heapq
import threading

class PriorityQueue(object):
    """
    Simple threadsafe heap based on the python heapq module.
    """
    def __init__(self):
        self._heap = []
        self._lock = threading.Lock()

    def push(self, item):
        with self._lock:
            heapq.heappush(self._heap, item)
    
    def pop(self):
        with self._lock:
            return heapq.heappop(self._heap)
    
    def __len__(self):
        return len(self._heap)

class FifoQueue(object):
    """
    Simple FIFO queue based on collections.deque.
    """
    def __init__(self):
        self._deque = collections.deque() # Documentation says this is threadsafe for push and pop

    def push(self, item):
        self._deque.append(item)
    
    def pop(self):
        return self._deque.popleft()
    
    def __len__(self):
        return len(self._deque)

class LifoQueue(object):
    """
    Simple LIFO queue based on collections.deque.
    """
    def __init__(self):
        self._deque = collections.deque() # Documentation says this is threadsafe for push and pop

    def push(self, item):
        self._deque.append(item)
    
    def pop(self):
        return self._deque.pop()
    
    def __len__(self):
        return len(self._deque)
    
class ThreadPool(object):
    """
    Manages a set of worker threads and dispatches tasks to them.
    """

    #_DefaultQueueType = FifoQueue
    #_DefaultQueueType = LifoQueue
    _DefaultQueueType = PriorityQueue
    
    def __init__(self, num_workers, queue_type=_DefaultQueueType):
        """
        Constructor.  Starts all workers.
        
        :param num_workers: The number of worker threads to create.
        :param queue_type: The type of queue to use for prioritizing tasks.  Possible queue types include :py:class:`PriorityQueue`,
                           :py:class:`FifoQueue`, and :py:class:`LifoQueue`, or any class with ``push()``, ``pop()``, and ``__len__()`` methods.
        """
        self.job_condition = threading.Condition()
        self.unassigned_tasks = queue_type()

        self.workers = self._start_workers( num_workers, queue_type )

        # ThreadPools automatically stop upon program exit
        atexit.register( self.stop )

    def wake_up(self, task):
        """
        Schedule the given task on the worker that is assigned to it.
        If it has no assigned worker yet, assign it to the first worker that becomes available.
        """
        # Once a task has been assigned, it must always be processed in the same worker
        if hasattr(task, 'assigned_worker') and task.assigned_worker is not None:
            task.assigned_worker.wake_up( task )
        else:
            self.unassigned_tasks.push(task)
            # Notify all currently waiting workers that there's new work
            self._notify_all_workers()

    def stop(self):
        """
        Stop all threads in the pool, and block for them to complete.
        Postcondition: All worker threads have stopped.  Unfinished tasks are simply dropped.
        """
        for w in self.workers:
            w.stop()
        
        for w in self.workers:
            w.join()
    
    def _start_workers(self, num_workers, queue_type):
        """
        Start a set of workers and return the set.
        """
        workers = set()
        for i in range(num_workers):
            w = _Worker(self, i, queue_type=queue_type)
            workers.add( w )
            w.start()
        return workers

    def _notify_all_workers(self):
        """
        Wake up all worker threads that are currently waiting for work.
        """
        for worker in self.workers:
            with worker.job_queue_condition:
                worker.job_queue_condition.notify()

class _Worker(threading.Thread):
    """
    Runs in a loop until stopped.
    The loop pops one task from the threadpool and executes it.
    """

    def __init__(self, thread_pool, index, queue_type ):
        name = "Worker #{}".format(index)
        super(_Worker, self).__init__( name=name )
        self.daemon = True # kill automatically on application exit!
        self.thread_pool = thread_pool
        self.stopped = False
        self.job_queue_condition = threading.Condition()
        self.job_queue = queue_type()
        
    def run(self):
        """
        Keep executing available tasks until we're stopped.
        """
        # Try to get some work.
        next_task = self._get_next_job()

        while not self.stopped:
            # Start (or resume) the work by switching to its greenlet
            next_task()

            # Try to get some work.
            next_task = self._get_next_job()

    def stop(self):
        """
        Tell this worker to stop running.
        Does not block for thread completion.
        """
        self.stopped = True
        # Wake up the thread if it's waiting for work
        with self.job_queue_condition:
            self.job_queue_condition.notify()

    def wake_up(self, task):
        """
        Add this task to the queue of tasks that are ready to be processed.
        The task may or not be started already.
        """
        assert task.assigned_worker is self
        with self.job_queue_condition:
            self.job_queue.push(task)
            self.job_queue_condition.notify()

    def _get_next_job(self):
        """
        Get the next available job to perform.
        If necessary, block until:
            - a task is available (return it) OR
            - the worker has been stopped (might return None)
        """
        # Keep trying until we get a job        
        with self.job_queue_condition:
            next_task = self._pop_job()

            while next_task is None and not self.stopped:
                # Wait for work to become available
                self.job_queue_condition.wait()
                next_task = self._pop_job()

        if not self.stopped:
            assert next_task is not None
            assert next_task.assigned_worker is self

        return next_task
    
    def _pop_job(self):
        """
        Non-blocking.
        If possible, get a job from our own job queue.
        Otherwise, get one from the global job queue.
        Return None if neither queue has work to do.
        """
        # Try our own queue first
        if len(self.job_queue) > 0:
            return self.job_queue.pop()

        # Otherwise, try to claim a job from the global unassigned list            
        try:
            task = self.thread_pool.unassigned_tasks.pop()
        except IndexError:
            return None
        else:
            task.assigned_worker = self # If this fails, then your callable is some built-in that doesn't allow arbitrary  
                                        #  members (e.g. .assigned_worker) to be "monkey-patched" onto it.  You may have to wrap it in a custom class first.
            return task
    
