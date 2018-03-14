//#include "testCRF.cpp"
#include <iostream>
#include <string>
#include <map>
#include <math.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include "crf.cpp"
using namespace CRFModel;
using namespace std;

class Test
{
    public:
    int a = 10;
    static int b;
    double *const c;
    Test()
    :c(new double)
    {
    }
    int getB()
    {
        return b;
    }
};
int Test::b = 100;

const int s = 11;

int getInt(int a, int b = s)
{
    return b;
}
auto makeTuple()
{
    return make_tuple("zliag", 12, 12.3, "sdfff");
}
void testPotential()
{
    for(int i = 0; i < 100; ++i)
    {
        std::unique_ptr<Matrix> message(new Matrix(10, Vector(100, 0.0)));
        //PotentialTable *p = new PotentialTable(100, 10, 100, 100);
        //std::unique_ptr<PotentialTable> p(new PotentialTable(100, 10, 100, 100));
        cout << "sleep for 3 sec..." << endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}
int main(void)
{
    //testPotential();

    cout << pow(12, 2) << endl;

    Test t;
    cout << t.a << endl;
    cout << t.getB() << endl;
    cout << getInt(1) << endl;
    map<string, int> m;
    m.insert(pair<string, int>("liuzhuang", 129));
    cout << m.size() << endl;

    int *const b = new int;
    delete b;

    auto tup = makeTuple();
    cout << std::get<1>(tup) << endl;
    cout << std::get<2>(tup) << endl;
    vector<vector<vector<int> > > tensor;

    cout << log(1) << endl;

    vector<int> *r = new vector<int>(2, 21);
    cout << r->operator[](1) << endl;

    // for (int j = 0; j < 50; ++j)
    // {
    //     vector<int> * vec = new vector<int>(10000, 133);
    //     //std::unique_ptr<vector<int> > vec(new vector<int>(10000, 133));
    //     std::this_thread::sleep_for(std::chrono::seconds(3));
    //     cout << "new" << endl;
    // }
    cout << "DDDDDDDDDDDDDDDDDDDDDDDDD" << endl;
    for_each(r->begin(), r->end(), [](int &i){i+=2;});
    for(auto &i : *r)
    {
        cout << i << endl;
    }
    int atomicNumber = 100;
    std::atomic<bool> foo[atomicNumber];
    mutex mtx;
    cout << "atomic : " << sizeof(foo[2])  << endl;
    cout << "mutex  : " << sizeof(mtx) << endl;
    for(auto &a : foo)
    {
        if(a == false)
        {
            cout << " " << a << endl;
        }
    }
    cout <<"atomic flag size: " << sizeof(std::atomic_flag) << endl;

    size_t blockSize = 6458496;
    double *dataBlock = new double[blockSize];
    //dataBlock[1000] = 1000;
    memset(dataBlock, -1.3, blockSize);
    memset(dataBlock, 0.0, blockSize);
    for_each(dataBlock, dataBlock + 6458496, [](double i){cout << i << " ";});
    cout << endl;

}