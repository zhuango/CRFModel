#include "crfMulThread.hpp"
#include <iostream>
#include <string>
#include <map>
#include <math.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std;
using namespace CRFModel;

vector<string> split(string line, char delim)
{
    istringstream toLine(line);
    vector<string> tokens;
    string item;
    while(getline(toLine, item, delim))
    {
        tokens.push_back(item);
    }
    return tokens;
}

vector<Seq*> *load(string filename)
{
    vector<Seq*> *data = new vector<Seq*>();
    size_t maxLen = Seq::MaxLength;
    string line;
    ifstream file(filename.c_str());
    size_t index = 0;
    while(getline(file, line))
    {
        vector<string> *wordSeq = new vector<string>();
        VectorInt *labels = new VectorInt();
        vector<string> tokens = split(line, ' ');

        for(string &str : tokens)
        {
            if (str == "")
            {
                continue;
            }
            vector<string> word_label = split(str, '/');
            size_t pos = word_label[1].find(']');
            if (pos != std::string::npos)
            {
                word_label[1] = word_label[1].substr(0, pos);
            }
            wordSeq->push_back(word_label[0]);
            labels->push_back(Seq::LabelTable[word_label[1]]);
        }
        size_t currentSeqLength = wordSeq->size();
        if (currentSeqLength > maxLen)
        {
            maxLen = currentSeqLength;
        }
        data->push_back(new Seq(wordSeq, labels));
            
        // if((++index) == 100)
        // {
        //     Seq::MaxLength = maxLen;
        //     return data;
        // }
    }
    Seq::MaxLength = maxLen;
    return data;
}

int main()
{
    string fold = "4";
    string trainCorpus = "/home/laboratory/github/homeWork/machineTranslation/data/train.txt" + fold;
    string testCorpus  = "/home/laboratory/github/homeWork/machineTranslation/data/test.txt" + fold;
    cout << "cache label..." << endl;
    ifstream labels;
    labels.open("/home/laboratory/github/homeWork/machineTranslation/data/pos.txt");
    int index = 0;
    string line;
    while(getline(labels, line))
    {
        vector<string> tokens = split(line, ' ');
        istringstream ss(tokens[0]);
        ss >> index;
        Seq::LabelTable.insert(pair<string, int>(tokens[1], index));
    }
    Seq::LabelSizeSquare = pow(Seq::LabelTable.size(), 2);
    
    cout << "load train data..." << endl;
    vector<Seq*> *train = load(trainCorpus);

    cout << "load test data ..." << endl;
    vector<Seq*> *test = load(testCorpus);

    size_t nodeFeatureSize = Seq::LabelTable.size() * Seq::WordsTable.size() * 7;
    size_t edgeDeatureSize = 0;//Seq::LabelTable.size() * Seq::LabelTable.size() * Seq::WordsTable.size();
    Seq::FeatureOffset = Seq::LabelTable.size() * Seq::WordsTable.size();
    cout << "generating feature table..." << endl;
    int countd = 0;
    for(auto &elem : *train)
    {
        countd += 1;
        if (countd % 500 == 0)
        {
            cout << countd << endl;
        }
        elem->GeneFeaturetable();
    }
    cout << "generating feature table..." << endl;
    countd = 0;
    for(auto &elem : *test)
    {
        countd += 1;
        if (countd % 500 == 0)
        {
            cout << countd << endl;
        }
        elem->GeneFeaturetable();
    }
    cout << nodeFeatureSize << " " << edgeDeatureSize << endl;
    CRFBin crf(nodeFeatureSize, edgeDeatureSize, Seq::LabelTable.size());
    cout << "training..." << endl;
    crf.SGA(*train, 0.2, 50, 2, true, test);
    cout << "testing..." << endl;
    double correct = 0.0;
    double count = 0.0;
    string resultFile = "./resultfold" + fold;
    ofstream resultStream(resultFile);
    for(Seq *seq : *test)
    {
        std::unique_ptr<VectorInt> result(crf.Sample(*seq));
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
    resultStream.close();

    cout << "accuracy: " << (correct / count) << endl;
    return 0;
}
