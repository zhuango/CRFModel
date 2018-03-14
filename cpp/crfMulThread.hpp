#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <algorithm>
#include <math.h>
#include <cstring>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <fstream>
#include <sstream>
#include "ThreadManager.hpp"

using namespace std;

namespace CRFModel
{
    const int InvalidState = -1;
    typedef vector<int> VectorInt;
    typedef vector<double> Vector;
    typedef vector<vector<double> > Matrix;
    typedef vector<vector<int> > MatrixInt;

    class Seq
    {
        public:
            static map<string, int> LabelTable;
            static size_t MaxLength;
            static size_t LabelSizeSquare;
            static map<string, int> WordsTable;
            static size_t FeatureOffset;

            const vector<string> * const Sequence;
            const vector<int> *    const Labels;
            bool isDisposed = false;
            //vector<vector<vector<int> > > featureTable;
            vector<vector<int> > featureTable;
            
            Seq(vector<string> *s, VectorInt *labels)
            :Sequence(s), Labels(labels)
            {
                for(auto &word : *Sequence)
                {
                    if (WordsTable.find(word) == WordsTable.end())
                    {
                        WordsTable.insert(pair<string, int>(word, WordsTable.size()));
                    }
                }
            }
            ~Seq()
            {
                delete Sequence;
                delete Labels;
            }
            void GeneFeaturetable()
            {
                size_t labelsize = LabelTable.size();
                size_t index = 0;
                for(int i = 0; i < Sequence->size(); ++i)
                {
                    int seqNum = i;
                    //vector<vector<int> > matrix;
                    for(int y0State=0; y0State < labelsize; ++y0State)
                    {
                        vector<int> weightIndex;
                        
                        // %x[0,0]
                        index = seqNum * labelsize + y0State;
                        // %x[-1,0]
                        // %x[-1,0]/%x[0,0]
                        if (seqNum > 0)
                        {
                            weightIndex.push_back(1*FeatureOffset+WordsTable[(*Sequence)[seqNum-1]] * labelsize + y0State);
                            weightIndex.push_back(5*FeatureOffset+WordsTable[(*Sequence)[seqNum-1]] * labelsize + y0State);
                            weightIndex.push_back(5*FeatureOffset+WordsTable[(*Sequence)[seqNum]] * labelsize + y0State);
                        }
                        // %x[-2,0]
                        if (seqNum > 1)
                        {
                            weightIndex.push_back(2*FeatureOffset+WordsTable[(*Sequence)[seqNum-2]] * labelsize + y0State);
                        }
                        // %x[1,0]
                        // %x[0,0]/%x[1,0]
                        if (seqNum < size()-1)
                        {
                            weightIndex.push_back(3*FeatureOffset+WordsTable[(*Sequence)[seqNum+1]] * labelsize + y0State);
                            weightIndex.push_back(6*FeatureOffset+WordsTable[(*Sequence)[seqNum+1]] * labelsize + y0State);
                            weightIndex.push_back(6*FeatureOffset+WordsTable[(*Sequence)[seqNum]] * labelsize + y0State);
                        }
                        // %x[2,0]
                        if (seqNum < size()-2)
                        {
                            weightIndex.push_back(4*FeatureOffset+WordsTable[(*Sequence)[seqNum+2]] * labelsize + y0State);
                        }
                        weightIndex.push_back(index);

                        featureTable.push_back(weightIndex);
                    }
                    //featureTable.push_back(matrix);
                }
            }
            VectorInt& GetFeature(int seqNum, int y0State, int y1State = InvalidState)
            {
                //const string &word = (*Sequence)[seqNum];
                
                //int wordNum = WordsTable[word];
                //int index = 0;
                int labelsize = LabelTable.size();

                return featureTable[seqNum * labelsize + y0State];
            }
            int size()
            {
                return this->Sequence->size();
            }
    };
    size_t Seq::MaxLength = 0;
    size_t Seq::LabelSizeSquare = 0;
    map<string, int> Seq::LabelTable;
    map<string, int> Seq::WordsTable;
    size_t Seq::FeatureOffset = 0;


    class PotentialTable
    {
        public:
            double *const Log0;
            double *const Logs;
            const size_t Log0Size;
            const size_t LogsSize0;
            const size_t LogsSize1;
            const size_t LogsSize2;
            size_t LogsSize1MulLogSize2;
            PotentialTable(int log0Size, int logsSize0, int logsSize1, int logsSize2)
            :
            Log0Size(log0Size),
            LogsSize0(logsSize0),
            LogsSize1(logsSize1),
            LogsSize2(logsSize2),
            Log0(new double[log0Size]),
            Logs(new double[logsSize0 * logsSize1 * logsSize2])
            {
                LogsSize1MulLogSize2 = LogsSize1 * LogsSize2;
            }

            ~PotentialTable()
            {
                delete[] Log0;
                delete[] Logs;
            }      

    };

    class CRFBin
    {
        public:
            CRFBin(int nodeFeatureSize, int edgeFeatureSize, int labelStateSize)
            {
                mWnode = new double[nodeFeatureSize];
                mWedge = new double[edgeFeatureSize];
                mWnodeGradient = new double[nodeFeatureSize];
                mWedgeGradient = new double[edgeFeatureSize];
                memset(mWnode, 0.0, nodeFeatureSize);
                memset(mWedge, 0.0, edgeFeatureSize);

                mNodeFeatureSize = nodeFeatureSize;
                mEdgeFeatureSize = edgeFeatureSize;
                mLabelStateSize  = labelStateSize;

                mEdgeFlag = new atomic_flag[edgeFeatureSize];
                memset(mEdgeFlag, 0, edgeFeatureSize);
            }
            ~CRFBin()
            {
                delete[] mWnode;
                delete[] mWedge;
                delete[] mWnodeGradient;
                delete[] mWedgeGradient;
                delete[] mEdgeFlag;
            }
            PotentialTable *LogPotentialTable(Seq &sequence)
            {
                size_t seqLength = sequence.Sequence->size();
                unique_ptr<VectorInt> feature;
                PotentialTable *potentialTable = new PotentialTable(mLabelStateSize, seqLength-1, mLabelStateSize, mLabelStateSize);

                for(int i = 0; i < mLabelStateSize; ++i)
                {
                    potentialTable->Log0[i] = product(sequence.GetFeature(0, i));
                }

                ThreadManager threadManager;
                for(int i = 1; i < seqLength; ++i)
                {
                    threadManager.Run(new thread(
                        [&sequence, potentialTable, this](int i)
                        {
                            unique_ptr<VectorInt> feature;
                            int index = 0;
                            int jindex = 0;
                            double node = 0.0;
                            double edge = 0.0;
                            for(int j = 0; j < this->mLabelStateSize; ++j)
                            {
                                jindex = j * potentialTable->LogsSize2;
                                for(int k = 0; k < this->mLabelStateSize; ++k)
                                {
                                    node  = product(sequence.GetFeature(i, k));
                                    // edge  = product(sequence.GetFeature(i, j, k));
                                    
                                    index = (i-1) * potentialTable->LogsSize1MulLogSize2 + jindex + k;
                                    potentialTable->Logs[index] = node + edge;
                                }
                            }
                        },
                        i
                    ));
                }
                threadManager.Join();

                return potentialTable;
            }

            double Loglikelihood(Seq &sequence)
            {                    
                size_t seqLength = sequence.Sequence->size();
                const VectorInt seqStates = *(sequence.Labels);

                unique_ptr<PotentialTable> potentialTable(LogPotentialTable(sequence));
                unique_ptr<Matrix> messages(forward(*potentialTable, seqLength));

                double maxProb = *(max_element((*messages)[seqLength-1].begin(), (*messages)[seqLength-1].end()));
                double tempMargin = 0.0;
                for(int i = 0; i < mLabelStateSize; ++i)
                {
                    tempMargin += exp((*messages)[seqLength-1][i] - maxProb);
                }
                double logNormalizedTerm = log(tempMargin) + maxProb;
                double logLikelihood = potentialTable->Log0[seqStates[0]];
                int y1 = 0;
                int y2 = 0;
                int index = 0;
                for(int i = 1; i < seqLength; ++i)
                {
                    y1 = seqStates[i-1];
                    y2 = seqStates[i];
                    index = (i-1) * potentialTable->LogsSize1MulLogSize2 +
                            y1    * potentialTable->LogsSize2 + y2;
                    logLikelihood += potentialTable->Logs[index];
                }
                logLikelihood -= logNormalizedTerm;

                return logLikelihood;
            }

            VectorInt *Sample(Seq &sequence)
            {
                size_t seqLength = sequence.Sequence->size();
                int index = 0;
                double max = 0.0;
                int maxarg = 0;
                PotentialTable * potentialTable = LogPotentialTable(sequence);
                VectorInt *labels = new VectorInt(seqLength, 0);
                MatrixInt path(seqLength, VectorInt(mLabelStateSize, 0));
                double *tempPotential = new double[potentialTable->Log0Size];
                double *prevPotential = new double[potentialTable->Log0Size];
                double *currPotential = new double[potentialTable->Log0Size];
                double *tempForSwap = NULL;
                copy(potentialTable->Log0, potentialTable->Log0 + potentialTable->Log0Size, prevPotential);
                for(int i = 1; i < seqLength; ++i)
                {
                    for (int j = 0; j < mLabelStateSize; ++j)
                    {   
                        max = 0.0;
                        maxarg= 0;                     
                        index = (i-1) * potentialTable->LogsSize1MulLogSize2 + j;
                        for(int k = 0; k < mLabelStateSize; ++k)
                        {
                            tempPotential[k] = prevPotential[k] + potentialTable->Logs[index + k*potentialTable->LogsSize2];
                            if (tempPotential[k] > tempPotential[maxarg])
                            {
                                maxarg = k;
                                max = tempPotential[k];
                            }
                        }
                        path[i-1][j] = maxarg;
                        currPotential[j] = max;
                    }
                    tempForSwap = currPotential;
                    currPotential = prevPotential;
                    prevPotential = tempForSwap;
                }
                maxarg= 0;
                for(int i = 1; i < mLabelStateSize; ++i)
                {
                    if (prevPotential[i] > prevPotential[maxarg])
                    {
                        maxarg = i;
                    }
                }
                (*labels)[labels->size()-1] = maxarg;
                for(int i = labels->size()-2; i >= 0; --i)
                {
                    (*labels)[i] = path[i][(*labels)[i + 1]];
                }

                delete[] currPotential;
                delete[] prevPotential;
                delete[] tempPotential;
                delete potentialTable;

                return labels;
            }
            void SGA(vector<Seq*> &sequences, double threshold, size_t iterations=20, double a=1, bool validate=false, vector<Seq*> *testData=nullptr)
            {
                double oldLikelihood = -10000000000000000;
                size_t earlyStopCount = 3;
                size_t dataSize = sequences.size();
                for(int i = 0; i < iterations; ++i)
                {
                    double rate = a;// / (sqrt(i) + 1);
                    cout << "learning rate = " << rate << endl;
                    cout << "Iteration: " << i << endl;

                    //performance//////////////////////////////////////////
                    clock_t start = clock();
                    //performance//////////////////////////////////////////
                    for(int j = 0; j < dataSize; ++j)
                    {
                        update(*sequences[j], rate);
                    }

                    cout << "Cal loglikelihood.." << endl;

                    double likelihood = 0.0;
                    
                    // ThreadManager threadManager;
                    // mutex mtx;
                    // for(int j = 0; j < dataSize; ++j)
                    // {
                    //     threadManager.Run(new thread(
                    //         [&sequences, &mtx, &likelihood, this](int j)
                    //         {
                    //             double likelood = this->Loglikelihood(*sequences[j]);
                    //             mtx.lock();
                    //             likelihood += likelood;
                    //             mtx.unlock();
                    //         },
                    //         j
                    //     ));
                    // }
                    // threadManager.Join();

                    //performance//////////////////////////////////////////
                    clock_t calGredient = clock() - start;
                    cout << "update: " << float(calGredient)/CLOCKS_PER_SEC << "s" << endl;
                    //performance//////////////////////////////////////////

                    cout << "Loglikelihood: " << likelihood / dataSize << endl;
                    // if (likelihood <= oldLikelihood)
                    // {
                    //     earlyStopCount -= 1;
                    //     if (earlyStopCount == 0) return;
                    // }
                    // else
                    // {
                    //     earlyStopCount = 3;
                    // }

                    cout << "===========================================" << endl;
                    oldLikelihood = likelihood;                    
                    // if (abs(likelihood) < threshold)
                    // {
                    //     return ;
                    // }
                    if (validate)
                    {
                        test(*testData, i);
                    }
                }
            }
            void test(vector<Seq*> &test, int i)
            {
                cout << "testing..." << endl;
                double correct = 0.0;
                double count = 0.0;
                ostringstream ss;
                ss << i;
                string resultFile = "./result/result" + ss.str();
                ofstream resultStream(resultFile);
                //performance//////////////////////////////////////////
                clock_t start = clock();
                //performance//////////////////////////////////////////
                for(Seq *seq : test)
                {
                    std::unique_ptr<VectorInt> result(Sample(*seq));
                    // for_each(result->begin(), result->end(), [](int i){cout << i << " ";});
                    // cout << endl;
                    // for_each(seq->Labels->begin(), seq->Labels->end(), [](int i){cout << i << " ";});
                    // cout << endl;
                    for (int i = 0; i < seq->size(); ++i)
                    {
                        resultStream << (*result)[i] << "\t" << (*(seq->Labels))[i] << endl;
                        if((*result)[i] == (*(seq->Labels))[i])
                        {
                            correct += 1.0;
                        }
                        count += 1.0;
                    }
                }
                //performance//////////////////////////////////////////
                clock_t calGredient = clock() - start;
                cout << "testing : " << float(calGredient)/CLOCKS_PER_SEC << "s" << endl;
                //performance//////////////////////////////////////////
                resultStream.close();

                cout << "accuracy: " << (correct / count) << endl;
            }
        private:
            double *mWnode;
            double *mWedge;
            double *mWnodeGradient;
            double *mWedgeGradient;
            size_t  mLabelStateSize;
            size_t  mNodeFeatureSize;
            size_t  mEdgeFeatureSize;
            atomic_flag *mEdgeFlag;

            size_t testMulThread = 0;

            double product(VectorInt &indexVector, bool isNode=true)
            {
                double result = 0.0;
                double *w = NULL;
                if(isNode)
                {
                    w = mWnode;
                }
                else
                {
                    w = mWedge;
                }
                for(auto &index : indexVector)
                {
                    result += w[index];
                }
                return result;
            }
            Matrix *forward(PotentialTable &potentialTable, size_t seqLength)
            {
                double tempMeg  = 0.0;
                double temp     = 0.0;
                int index       = 0;
                int i,j,k;
                double maxProb  = 0.0;
                Vector Fs(mLabelStateSize, 0.0);

                Matrix *message = new Matrix(seqLength, Vector(mLabelStateSize, 0.0));
                copy(potentialTable.Log0, potentialTable.Log0 + mLabelStateSize, (*message)[0].begin());
                for(i = 1; i < seqLength; ++i)
                {
                    for(j = 0; j < mLabelStateSize; ++j)
                    {
                        tempMeg = 0.0;
                        index   = (i-1) * potentialTable.LogsSize1MulLogSize2 + j;

                        for(k = 0; k < mLabelStateSize; ++k)
                        {
                            Fs[k] = potentialTable.Logs[index + k*potentialTable.LogsSize2] + (*message)[i-1][k];
                        }
                        maxProb = *(max_element(Fs.begin(), Fs.end()));

                        for(k = 0; k < mLabelStateSize; ++k)
                        {
                            tempMeg += exp(Fs[k] - maxProb);
                        }
                        (*message)[i][j] = log(tempMeg) + maxProb;
                    }
                }
                return message;
            }
            Matrix *backward(PotentialTable &potentialTable, int seqLength)
            {
                double tempMeg  = 0.0;
                double temp     = 0.0;
                int index    = 0;
                int i,j,k, iindex;
                double maxProb  = 0.0;
                Vector Fs(mLabelStateSize, 0.0);

                Matrix *message = new Matrix(seqLength, Vector(mLabelStateSize, 0.0));
                for(i = seqLength-2; i >= 0; --i)
                {
                    iindex = i * potentialTable.LogsSize1MulLogSize2;
                    for(j = 0; j < mLabelStateSize; ++j)
                    {
                        tempMeg = 0.0;
                        index   = iindex + j * potentialTable.LogsSize2;

                        for(k = 0; k < mLabelStateSize; ++k)
                        {
                            Fs[k] = potentialTable.Logs[index + k] + (*message)[i+1][k];
                        }
                        maxProb = *(max_element(Fs.begin(), Fs.end()));

                        for(k = 0; k < mLabelStateSize; ++k)
                        {
                            tempMeg += exp(Fs[k] - maxProb);
                        }
                        (*message)[i][j] = log(tempMeg) + maxProb;

                    }
                }
                return message;
            }

            void calEdgeGradient(
                int i,
                PotentialTable *potentialTable,
                Seq &sequence,
                Matrix *forwardMessages,
                Matrix *backwardMessages,
                double logNormalizedTerm
            )
            {
                unique_ptr<VectorInt> feature5;
                int index = 0;
                int j, k, jindex;

                for(j = 0; j < mLabelStateSize; ++j)
                {
                    jindex = j * potentialTable->LogsSize2;
                    for(k = 0; k < mLabelStateSize; ++k)
                    {
                        index = (i-1) * potentialTable->LogsSize1MulLogSize2 + jindex + k;
                        for(auto &elem : sequence.GetFeature(i, j ,k))
                        {
                            // memory care.
                            while(mEdgeFlag[elem].test_and_set())
                            {
                                this_thread::yield();
                            }
                            mWedgeGradient[elem] += exp((*forwardMessages)[i-1][j] + potentialTable->Logs[index] + (*backwardMessages)[i][k] - logNormalizedTerm);
                            mEdgeFlag[elem].clear();
                        }
                    }
                }
            }
            void update(Seq &sequence, double rate)
            {
                size_t seqLength = sequence.Sequence->size();
                const vector<int> * const labels = sequence.Labels;
                int index = 0;
                int i,j,k;

                unique_ptr<PotentialTable> potentialTable(LogPotentialTable(sequence));
                unique_ptr<Matrix> forwardMessages;
                thread forwThr([&potentialTable, seqLength, &forwardMessages, this]()
                    {
                        forwardMessages.reset(this->forward(*potentialTable, seqLength));
                    }
                );
                unique_ptr<Matrix> backwardMessages;
                thread backThr([&potentialTable, seqLength, &backwardMessages, this]()
                    {
                        backwardMessages.reset(this->backward(*potentialTable, seqLength));
                    }
                );
                unique_ptr<VectorInt> feature;
                
                forwThr.join();
                double maxProb = *(max_element((*forwardMessages)[seqLength-1].begin(), (*forwardMessages)[seqLength-1].end()));
                double tempMargin = 0.0;
                for(i = 0; i < mLabelStateSize; ++i)
                {
                    tempMargin += exp((*forwardMessages)[seqLength-1][i] - maxProb);
                }
                double logNormalizedTerm = log(tempMargin) + maxProb;
                for_each(mWnodeGradient, mWnodeGradient + mNodeFeatureSize, [](double &i){i=0.0;});
                for_each(mWedgeGradient, mWedgeGradient + mEdgeFeatureSize, [](double &i){i=0.0;});
                // memset(mWnodeGradient, 0.0, mNodeFeatureSize);
                // memset(mWedgeGradient, 0.0, mEdgeFeatureSize);
                
                backThr.join();
                for(i = 0; i < seqLength; ++i)
                {
                    for(j = 0; j < mLabelStateSize; ++j)
                    {
                        for(auto &elem : sequence.GetFeature(i, j))
                        {
                            mWnodeGradient[elem] += exp((*forwardMessages)[i][j] + (*backwardMessages)[i][j] - logNormalizedTerm);
                        }
                    }
                }
    
                // ThreadManager threadManager;
                // for(i = 1; i < seqLength; ++i)
                // {
                //     threadManager.Run(new thread(
                //         [this, &potentialTable, &sequence, &forwardMessages, &backwardMessages, logNormalizedTerm](int i)
                //         {
                //             this->calEdgeGradient(
                //                 i,
                //                 potentialTable.get(),
                //                 ref(sequence),
                //                 forwardMessages.get(),
                //                 backwardMessages.get(),
                //                 logNormalizedTerm
                //             );
                //         },
                //         i
                //     ));
                // };
                // threadManager.Join();

                for(i = 0; i < mNodeFeatureSize; ++i)
                {
                    mWnode[i] -= mWnodeGradient[i] * rate;
                }
                for(i = 0; i < mEdgeFeatureSize; ++i)
                {
                    mWedge[i] -= mWedgeGradient[i] * rate;
                }

                for(i = 0; i < seqLength; ++i)
                {
                    for(auto &elem : sequence.GetFeature(i, (*labels)[i]))
                    {
                        mWnode[elem] += rate;
                    }
                }
                if (mEdgeFeatureSize == 0)
                {
                    return;
                }
                for(i = 1; i < seqLength; ++i)
                {
                    for(auto &elem : sequence.GetFeature(i, (*labels)[i-1], (*labels)[i]))
                    {
                        mWedge[elem] += rate;
                    }
                }
            }
            
    };
}
