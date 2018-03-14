#include "crf.cpp"
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
    size_t maxLen = 0;
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
        
        if((++index) == 30)
        {
            Seq::MaxLength = maxLen;
            return data;
        }
    }
    Seq::MaxLength = maxLen;
    return data;
}

int main()
{
    string trainCorpus = "/home/laboratory/github/homeWork/machineTranslation/data/train.txt";
    string testCorpus  = "/home/laboratory/github/homeWork/machineTranslation/data/test.txt";
    cout << "cache label..." << endl;
    ifstream labels;
    labels.open("/home/laboratory/github/homeWork/machineTranslation/data/label.1.txt");
    int index = 0;
    string line;
    while(getline(labels, line))
    {
        vector<string> tokens = split(line, ' ');
        istringstream ss(tokens[0]);
        ss >> index;
        Seq::LabelTable.insert(pair<string, int>(tokens[1], index));
    }
    
    cout << "load train data..." << endl;
    vector<Seq*> *train = load("/home/laboratory/github/homeWork/machineTranslation/data/train.txt");

    cout << "load test data ..." << endl;
    vector<Seq*> *test = load("/home/laboratory/github/homeWork/machineTranslation/data/test.txt");

    size_t nodeFeatureSize = Seq::LabelTable.size() * Seq::WordsTable.size();
    size_t edgeDeatureSize = Seq::LabelTable.size() * Seq::LabelTable.size() * Seq::WordsTable.size();


    CRFBin crf(nodeFeatureSize, edgeDeatureSize, Seq::LabelTable.size());
    cout << "training..." << endl;
    crf.SGA(*train, 5, 1);
    cout << "testing..." << endl;
    for(Seq *seq : *test)
    {
        std::unique_ptr<VectorInt> result(crf.Sample(*seq));
        for_each(result->begin(), result->end(), [](int i){cout << i << " ";});
        cout << endl;
        for_each(seq->Labels->begin(), seq->Labels->end(), [](int i){cout << i << " ";});
        cout << endl;
    }

    return 0;
}