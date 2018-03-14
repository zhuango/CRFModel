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
//for test
#include <thread>

using namespace std;

namespace CRFModel
{
    const int InvalidState = -1;
    typedef vector<int> VectorInt;
    typedef vector<double> Vector;
    typedef std::unique_ptr<vector<int> > SmartVectorIntPointer;
    typedef vector<vector<double> > Matrix;
    typedef vector<vector<int> > MatrixInt;

    class Seq
    {
        public:
            static map<string, int> LabelTable;
            static map<string, int> WordsTable;
            static size_t MaxLength;

            const vector<string> * const Sequence;
            const vector<int> *    const Labels;
            bool isDisposed = false;
            
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
            VectorInt* GetFeature(int seqNum, int y0State, int y1State = InvalidState)
            {
                const string &word = (*Sequence)[seqNum];
                VectorInt *weightIndex = new VectorInt();
                int index = 0;
                int labelsize = LabelTable.size();
                
                if (InvalidState == y1State)
                {
                    index = WordsTable[word] * labelsize + y0State;
                }
                else
                {
                    index = WordsTable[word] * pow(labelsize, 2) + y0State * labelsize + y1State;
                }
                weightIndex->push_back(index);
                
                return weightIndex;
            }
    };
    size_t Seq::MaxLength = 0;
    map<string, int> Seq::LabelTable;
    map<string, int> Seq::WordsTable;


    class PotentialTable
    {
        public:
            double *const Log0;
            double *const Logs;
            const size_t Log0Size;
            const size_t LogsSize0;
            const size_t LogsSize1;
            const size_t LogsSize2;

            PotentialTable(int log0Size, int logsSize0, int logsSize1, int logsSize2)
            :
            Log0Size(log0Size),
            LogsSize0(logsSize0),
            LogsSize1(logsSize1),
            LogsSize2(logsSize2),
            Log0(new double[log0Size]),
            Logs(new double[logsSize0 * logsSize1 * logsSize2])
            {
                memset(Log0, 0, log0Size);
                memset(Logs, 0, logsSize0 * logsSize1 * logsSize2);
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
            }
            ~CRFBin()
            {
                delete[] mWnode;
                delete[] mWedge;
                delete[] mWnodeGradient;
                delete[] mWedgeGradient;
            }
            PotentialTable *LogPotentialTable(Seq &sequence)
            {
                int    index = 0;
                double edge  = 0;
                double node  = 0;
                size_t seqLength = sequence.Sequence->size();
                std::unique_ptr<VectorInt> feature;
                PotentialTable *potentialTable = new PotentialTable(mLabelStateSize, seqLength-1, mLabelStateSize, mLabelStateSize);

                for(int i = 0; i < mLabelStateSize; ++i)
                {
                    feature.reset(sequence.GetFeature(0, i));
                    potentialTable->Log0[i] = product(*feature);
                }
                for(int i = 1; i < seqLength; ++i)
                {
                    for(int j = 0; j < mLabelStateSize; ++j)
                    {
                        for(int k = 0; k < mLabelStateSize; ++k)
                        {
                            feature.reset(sequence.GetFeature(i, k));
                            node  = product(*feature);
                            feature.reset(sequence.GetFeature(i, j, k));
                            edge  = product(*feature, false);
                            
                            index = (i-1) * potentialTable->LogsSize1 * potentialTable->LogsSize2 +
                                    j     * potentialTable->LogsSize2 + k;
                            potentialTable->Logs[index] = node + edge;
                        }
                    }
                }
                return potentialTable;
            }

            double Loglikelihood(Seq &sequence)
            {
                size_t seqLength = sequence.Sequence->size();
                const VectorInt seqStates = *(sequence.Labels);

                std::unique_ptr<PotentialTable> potentialTable(LogPotentialTable(sequence));
                std::unique_ptr<Matrix> messages(forward(*potentialTable, seqLength));
                
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
                    index = (i-1) * potentialTable->LogsSize1 * potentialTable->LogsSize2 +
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
                        index = (i-1) * potentialTable->LogsSize1 * potentialTable->LogsSize2 + j;
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
            void SGA(vector<Seq*> &sequences, size_t iterations=20, double a0=1)
            {
                double oldLikelihood = -10000000000000000;
                size_t earlyStopCount = 3;
                size_t dataSize = sequences.size();
                for(int i = 0; i < iterations; ++i)
                {
                    double rate = a0 / (sqrt(i) + 1);
                    cout << "rate = " << rate << endl;
                    cout << "Iteration: " << i << endl;

                    //performance/////////
                    clock_t start = clock();
                    //performance/////////
                        
                    for(int j = 0; j < dataSize; ++j)
                    {
                        update(*sequences[j], rate);
                    }
                    
                    cout << "Cal loglikehood.." << endl;
                    double likelihood = 0.0;
                    for(int j = 0; j < dataSize; ++j)
                    {
                        likelihood += Loglikelihood(*sequences[j]);
                    }
                    cout << "Loglihood: " << likelihood / dataSize << endl;
                    if (likelihood <= oldLikelihood)
                    {
                        earlyStopCount -= 1;
                        if (earlyStopCount == 0) return;
                    }
                    else
                    {
                        earlyStopCount = 3;
                    }

                    //performance/////////
                    clock_t calGredient = clock() - start;
                    cout << "update: " << float(calGredient)/CLOCKS_PER_SEC << endl;
                    calGredient = clock();
                    //performance/////////
                    cout << "===========================================" << endl;

                    oldLikelihood = likelihood;
                    if (abs(likelihood) < 0.01)
                    {
                        return ;
                    }
                }
            }
        private:
            double *mWnode;
            double *mWedge;
            double *mWnodeGradient;
            double *mWedgeGradient;
            size_t  mLabelStateSize;
            size_t  mNodeFeatureSize;
            size_t  mEdgeFeatureSize;


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
                int index    = 0;
                double maxProb  = 0.0;
                Vector Fs(mLabelStateSize, 0.0);

                Matrix *message = new Matrix(seqLength, Vector(mLabelStateSize, 0.0));
                copy(potentialTable.Log0, potentialTable.Log0 + mLabelStateSize, (*message)[0].begin());
                for(int i = 1; i < seqLength; ++i)
                {
                    for(int j = 0; j < mLabelStateSize; ++j)
                    {
                        tempMeg = 0.0;
                        index   = (i-1) * potentialTable.LogsSize1 * potentialTable.LogsSize2 + j;

                        for(int k = 0; k < mLabelStateSize; ++k)
                        {
                            Fs[k] = potentialTable.Logs[index + k*potentialTable.LogsSize2] + (*message)[i-1][k];
                        }
                        maxProb = *(max_element(Fs.begin(), Fs.end()));

                        for(int k = 0; k < mLabelStateSize; ++k)
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
                double maxProb  = 0.0;
                Vector Fs(mLabelStateSize, 0.0);

                Matrix *message = new Matrix(seqLength, Vector(mLabelStateSize, 0.0));
                for(int i = seqLength-2; i >= 0; --i)
                {
                    for(int j = 0; j < mLabelStateSize; ++j)
                    {
                        tempMeg = 0.0;
                        index   = i * potentialTable.LogsSize1 * potentialTable.LogsSize2 + 
                                  j * potentialTable.LogsSize2;

                        for(int k = 0; k < mLabelStateSize; ++k)
                        {
                            Fs[k] = potentialTable.Logs[index + k] + (*message)[i+1][k];
                        }
                        maxProb = *(max_element(Fs.begin(), Fs.end()));

                        for(int k = 0; k < mLabelStateSize; ++k)
                        {
                            tempMeg += exp(Fs[k] - maxProb);
                        }
                        (*message)[i][j] = log(tempMeg) + maxProb;

                    }
                }
                return message;
            }

            void update(Seq &sequence, double rate)
            {
                size_t seqLength = sequence.Sequence->size();
                const vector<int> * const labels = sequence.Labels;
                int index = 0;

                std::unique_ptr<PotentialTable> potentialTable(LogPotentialTable(sequence));
                std::unique_ptr<Matrix> forwardMessages(forward(*potentialTable, seqLength));
                std::unique_ptr<Matrix> backwardMessages(backward(*potentialTable, seqLength));
                std::unique_ptr<VectorInt> feature;

                double maxProb = *(max_element((*forwardMessages)[seqLength-1].begin(), (*forwardMessages)[seqLength-1].end()));
                double tempMargin = 0.0;
                for(int i = 0; i < mLabelStateSize; ++i)
                {
                    tempMargin += exp((*forwardMessages)[seqLength-1][i] - maxProb);
                }
                double logNormalizedTerm = log(tempMargin) + maxProb;

                for_each(mWnodeGradient, mWnodeGradient + mNodeFeatureSize, [](double &i){i=0.0;});
                for_each(mWedgeGradient, mWedgeGradient + mEdgeFeatureSize, [](double &i){i=0.0;});

                for(int i = 0; i < seqLength; ++i)
                {
                    for(int j = 0; j < mLabelStateSize; ++j)
                    {
                        feature.reset(sequence.GetFeature(i, j));
                        for(auto &elem : *feature)
                        {
                            mWnodeGradient[elem] += exp((*forwardMessages)[i][j] + (*backwardMessages)[i][j] - logNormalizedTerm);
                        }
                    }
                }

                for(int i = 1; i < seqLength; ++i)
                {
                    for(int j = 0; j < mLabelStateSize; ++j)
                    {
                        for(int k = 0; k < mLabelStateSize; ++k)
                        {
                            index = (i-1) * potentialTable->LogsSize1 * potentialTable->LogsSize2 +
                                    j     * potentialTable->LogsSize2 + k;
                            feature.reset(sequence.GetFeature(i, j ,k));
                            for(auto &elem : *feature)
                            {
                                mWedgeGradient[elem] += exp((*forwardMessages)[i-1][j] + potentialTable->Logs[index] + (*backwardMessages)[i][k] - logNormalizedTerm);
                            }
                        }
                    }
                }
                for(int i = 0; i < mNodeFeatureSize; ++i)
                {
                    mWnode[i] -= mWnodeGradient[i] * rate;
                }
                for(int i = 0; i < mEdgeFeatureSize; ++i)
                {
                    mWedge[i] -= mWedgeGradient[i] * rate;
                }

                for(int i = 0; i < seqLength; ++i)
                {
                    feature.reset(sequence.GetFeature(i, (*labels)[i]));
                    for(auto &elem : *feature)
                    {
                        mWnode[elem] += rate;
                    }
                }
                for(int i = 1; i < seqLength; ++i)
                {
                    feature.reset(sequence.GetFeature(i, (*labels)[i-1], (*labels)[i]));
                    for(auto &elem : *feature)
                    {
                        mWedge[elem] += rate;
                    }
                }
            }
            
    };
}
