#include <iostream>
#include <thread>
#include <string>
#include "ThreadManager.hpp"
using namespace CRFModel;
using namespace std;
void threadFunc(int a, double b, string s)
{
    cout << a << " " << b << " " << s << endl;
}
int main(void)
{
    for(int i = 0; i < 100; i++)
    {
        ThreadManager thrs;
        thrs.Run(new thread(threadFunc, 1, 2.3, "s44444df"));
        thrs.Run(new thread(threadFunc, 2, 2.4, "s4df"));
        thrs.Run(new thread(threadFunc, 3, 2.5, "s56666df"));
        thrs.Run(new thread(threadFunc, 4, 2.6, "sd65f"));

        thrs.Join();

        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}