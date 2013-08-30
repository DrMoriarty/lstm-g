/*

  C++ class for building and using LSTM-g neural networks
  (c) 2013 Vasiliy Makarov, <drmoriarty.0@gmail.com>

  used original code on python from https://github.com/MrMormon/lstm-g
 */

#include <string>
#include <queue>
#include <vector>
#include <map>
#include <list>
#include <sstream>
#include <random>
#include <math.h>
#include <stdlib.h>

typedef double ntype;
using std::vector;
using std::map;
using std::string;
using std::pair;
using std::list;
using std::stringstream;

typedef pair<int, int> pkey;
typedef map<pkey, int> pmap;
typedef map<pkey, ntype> nmap;

class tkey : public pair<pkey, int> {
public:
  tkey(int a, int b, int c): pair<pkey, int>(pkey(a, b), c) {}
  tkey(pkey p, int c): pair<pkey, int>(p, c) {}
  int a() {
    return pair<pkey, int>::first.first;
  }
  int b() {
    return pair<pkey, int>::first.second;
  }
  int c() {
    return pair<pkey, int>::second;
  }
};

typedef map<tkey, ntype> tmap;

class LSTM_g {

 public:
  LSTM_g(string& specData);

  string toString(bool newNetwork, char newLine='\n');
  vector<ntype> step(vector<ntype>& inputs);
  ntype getError(vector<ntype>& targets);
  void learn(vector<ntype>& targets, ntype learningRate = 0.1);

 protected:  
  ntype actFunc(ntype s, bool derivative, ntype bias=0.0);
  ntype gain(int j, int i);
  ntype theTerm(int j, int k);
  void  build(vector<vector<string> >& specData);
  void  toLowLevel(vector<vector<string> >& specData);
  void  addConnection(vector<vector<string> >& specData, int j, int i, int g = -1);
  vector<int> unitsInBlock(int blockNum);

 private:

  pmap gater;
  vector<ntype> activation;
  vector<vector<ntype> > oldActivation;
  map<pkey, ntype > weight;
  vector<ntype> oldState;
  vector<ntype> state;
  nmap trace;
  tmap extendedTrace;
  map<int, int> layerData;

  int numInputs, numOutputs, numUnits, inputToOutput, biasOutput;
};
