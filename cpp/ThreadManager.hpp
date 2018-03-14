#pragma once

#include <thread>
#include <vector>
using namespace std;

namespace CRFModel
{
    class ThreadManager
    {
        public:
            ThreadManager()
            {
            }
            ~ThreadManager()
            {
                Join();
            }
            
            void Run(thread *thr)
            {
                mThreads.push_back(thr);
            }

            void Join()
            {
                for(auto *thr : mThreads)
                {
                    thr->join();
                }
                for(auto *thr : mThreads)
                {
                    delete thr;
                }
                mThreads.clear();
            }
        private:
            vector<thread*> mThreads;
    };

}