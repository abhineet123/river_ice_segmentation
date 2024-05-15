#pragma once
#include <list>
#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#include <sys/time.h>
#endif

namespace ANNThreading{

	//int TestAndSetBit(int* val, int bit);

	//int TestAndResetBit(int* val, int bit);


	class Event
	{
	private:
#ifdef _WIN32
		HANDLE m_hEvent;
#else
		pthread_cond_t m_condition_cond;
		pthread_mutex_t m_Mutex;
#endif
	public:
		Event(void)
		{
#ifdef _WIN32
			m_hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
#else
			pthread_mutex_init(&m_Mutex, NULL);
			pthread_cond_init(&m_condition_cond, NULL);
//			m_condition_cond = PTHREAD_COND_INITIALIZER;
#endif
		}
		~Event(void)
		{
#ifdef _WIN32
			CloseHandle(m_hEvent);
#else
			pthread_mutex_destroy(&m_Mutex);
			pthread_cond_destroy(&m_condition_cond);
#endif
		}

		void Wait(unsigned long timeout
#ifdef _WIN32
			= INFINITE
#endif
			)
		{
#ifdef _WIN32
			WaitForSingleObject(m_hEvent, timeout );
#else
			int               rc;
			struct timespec   ts;
			struct timeval    tp;
			rc =  gettimeofday(&tp, NULL);
			ts.tv_sec  = tp.tv_sec;
			ts.tv_nsec = tp.tv_usec * 1000 + timeout;
			ts.tv_sec += ts.tv_nsec/1000000000;
			ts.tv_nsec = ts.tv_nsec%1000000000;

			pthread_mutex_lock( &m_Mutex);
			pthread_cond_timedwait( &m_condition_cond, &m_Mutex, &ts);
			pthread_mutex_unlock( &m_Mutex );
#endif
		}
		void Pulse()
		{
#ifdef _WIN32
			SetEvent(m_hEvent);
#else
			pthread_cond_signal( &m_condition_cond );
#endif
		}
	};
	
	class Mutex
	{
#ifdef _WIN32
		CRITICAL_SECTION m_critSection;
#else 
		pthread_mutex_t m_Mutex;
#endif

	public:
		Mutex()
		{
#ifdef _WIN32
			InitializeCriticalSection(&m_critSection);
#else
			pthread_mutex_init(&m_Mutex, NULL);
#endif
		}
		~Mutex() 
		{
#ifdef _WIN32
			DeleteCriticalSection(&m_critSection);
#else
			pthread_mutex_destroy(&m_Mutex);
#endif
		}

		void Acquire ()
		{
#ifdef _WIN32
			EnterCriticalSection (&m_critSection);
#else
			pthread_mutex_lock(&m_Mutex);
#endif
		}
		void Release ()
		{
#ifdef _WIN32
			LeaveCriticalSection (&m_critSection);
#else
			pthread_mutex_unlock(&m_Mutex);
#endif
		}
	};

	class IRunnable {
	public:
		virtual void Run() = 0;
		virtual ~IRunnable(){};
	};

	

	class Thread
	{
	private:
		bool m_started;
#ifdef _WIN32
		HANDLE m_threadHandle;
#else
		pthread_t m_threadHandle;
#endif
		IRunnable* m_runnable;

	public:
		Thread(IRunnable* runnable = NULL) 
		{
			m_runnable = runnable;
			m_started = false;
			m_threadHandle = 0;
		}

		~Thread()
		{
#ifdef _WIN32
			if(m_threadHandle != 0){CloseHandle(m_threadHandle);}
#endif
		}
		void Start(IRunnable* runnable = NULL) {
			if(m_started)
			{
//				throw ThreadException("Thread already started.", this);
			}

			if(runnable != NULL)
			{
				m_runnable = runnable;
			}

			if(m_runnable == NULL)
			{
			//	throw ThreadException("An object implementing the IRunnable interface required.", this);
			}


#ifdef _WIN32
			DWORD threadID=0;
			m_threadHandle = CreateThread(0, 0, ThreadProc, this, 0, &threadID);
			if(m_threadHandle == 0)
			{
			//	throw ThreadException(GetLastError(), this);
			}
			Sleep(0);
#else
			int threadID;
			threadID = pthread_create( &m_threadHandle, NULL, ThreadProc,(void*) this);
#endif
		}

		void Join(unsigned long timeOut
#ifdef _WIN32
			=INFINITE
#endif
			) 
		{
			if(m_threadHandle != 0 && m_started)
			{
#ifdef _WIN32
				DWORD waitResult = ::WaitForSingleObject(m_threadHandle, timeOut);
				if(waitResult == WAIT_FAILED)
				{
				//	throw ThreadException(::GetLastError(), this);
				}
#else

				pthread_join(m_threadHandle, NULL);
#endif
			}

		}

		bool IsAlive() { return m_started; }

	protected:
		void run() 
		{
			m_started = true;
			m_runnable->Run();
			m_started = false;
		}
#ifdef _WIN32
		static unsigned long __stdcall ThreadProc(void* ptr) {
			((Thread *)ptr)->run();
			return 0;
		}
#else
		static void *ThreadProc( void *ptr )
		{
			((Thread *)ptr)->run();
			return NULL;
		}
#endif
	};

	class ITask
	{
	public:
		virtual void Run()=0;
		virtual ~ITask(){}
	};


	class TaskManager : public IRunnable
	{
	public:
		class TaskManagerListener
		{
		public:
			virtual void TaskFinished(ITask* task)=0;
			virtual ~TaskManagerListener(){}
		};
	private:
		bool m_IsDisposing;
		int m_NumberOfThreads;
		std::list<ITask*> m_Tasks;
		Mutex m_Mutex;
		Event m_NewTaskEvent;
		int m_TaskCount;
		std::list<Thread*> m_Threads;
		int m_AliveThreads;

		Event m_TaskFinishedEvent;
		TaskManagerListener* m_TaskManagerListener;
	public:

		TaskManager(int numberOfThreads = -1)
		{
			if(numberOfThreads < 1)
			{
#ifdef _WIN32
				SYSTEM_INFO info;
				GetSystemInfo(&info);
				numberOfThreads = info.dwNumberOfProcessors;
#else
				numberOfThreads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
				if(numberOfThreads < 1)numberOfThreads =1;
				//printf("Number Of Threads: %d\n", numberOfThreads);
			}

			m_AliveThreads = 0;
			m_TaskManagerListener = NULL;
			m_TaskCount = 0;
			m_NumberOfThreads = numberOfThreads;
			m_IsDisposing = false;

			for(int i = 0 ; i < m_NumberOfThreads ; i++)
			{
				Thread* t = new Thread(this);
				t->Start();
				m_Threads.push_back(t);
			}
			this->m_Mutex.Acquire();
			while(m_AliveThreads < m_NumberOfThreads)
			{
				this->m_Mutex.Release();
#ifdef _WIN32
				Sleep(0);
#else
				struct timespec timeOut,remains;timeOut.tv_sec = 0;timeOut.tv_nsec = 1;nanosleep(&timeOut, &remains);
#endif
				this->m_Mutex.Acquire();
			}
			this->m_Mutex.Release();
			
		}
		~TaskManager(void)
		{
			m_IsDisposing = true;
			m_Mutex.Acquire();
			for(std::list<ITask*>::iterator it = m_Tasks.begin() ; it != m_Tasks.end() ; it++)
			{
				delete *it;
			}
			m_Mutex.Release();
			for(std::list<Thread*>::iterator it = m_Threads.begin() ; it != m_Threads.end() ; it++)
			{
				m_NewTaskEvent.Pulse();
			}
			for(std::list<Thread*>::iterator it = m_Threads.begin() ; it != m_Threads.end() ; it++)
			{
				if((*it)->IsAlive())
				{
					(*it)->Join(1000);
				}
				delete (*it);
			}
		}

		int GetNumberOfThreads()
		{
			return m_NumberOfThreads;
		}

		void SetTaskManagerListener(TaskManagerListener* taskManagerListener)
		{
			m_TaskManagerListener = taskManagerListener;
		}

		void ScheduleTask(ITask* task)
		{
			m_Mutex.Acquire();
			m_TaskCount++;
			m_Tasks.push_back(task);
			m_Mutex.Release();
			m_NewTaskEvent.Pulse();
		}

		virtual void Run()
		{
			m_Mutex.Acquire();
			m_AliveThreads++;
			m_Mutex.Release();
			while(!m_IsDisposing)
			{
				ITask* task = NULL;
				m_Mutex.Acquire();
				if(m_Tasks.size() > 0)
				{
					task = m_Tasks.front();
					m_Tasks.pop_front();
				}
				m_Mutex.Release();
				if(task != NULL)
				{
					task->Run();
					TaskFinished(task);
					delete task;
					m_Mutex.Acquire();
					m_TaskCount--;
					m_Mutex.Release();
					m_TaskFinishedEvent.Pulse();
				}
				else
				{
					if(!m_IsDisposing)
					{
						m_NewTaskEvent.Wait(200);
					}
				}
			}

			m_Mutex.Acquire();
			m_AliveThreads--;
			m_Mutex.Release();

		}

		void TaskFinished(ITask* task)
		{
			if(m_TaskManagerListener != NULL)
			{
				m_TaskManagerListener->TaskFinished(task);
			}
		}

		void WaitAll()
		{
			while(!m_IsDisposing)
			{
				bool wait = false;
				m_Mutex.Acquire();
				wait = (m_TaskCount > 0);
				m_Mutex.Release();


				if(!wait)
				{
					break;
				}
				m_TaskFinishedEvent.Wait(1000);
			}
		}
	};
};
