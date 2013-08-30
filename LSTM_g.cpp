/*

  C++ class for building and using LSTM-g neural networks
  (c) 2013 Vasiliy Makarov, <drmoriarty.0@gmail.com>

  used original code on python from https://github.com/MrMormon/lstm-g
 */

#include "LSTM_g.hpp"

// main

#ifndef LIBRARY

int main(int argc, char** argv)
{
  return 0;
}

#endif

// public interface

LSTM_g::LSTM_g(std::string& specData)
{
}

std::string LSTM_g::toString(bool newNetwork, char newLine)
{
}

std::vector<ntype> LSTM_g::step(std::vector<ntype>& inputs)
{
}

ntype LSTM_g::getError(std::vector<ntype>& targets)
{
}

void LSTM_g::learn(std::vector<ntype>& targets, ntype learningRate)
{
}

// private funcitons

ntype LSTM_g::actFunc(ntype s, bool derivative, ntype bias)
{
  ntype value = 1.0 / (1.0 + exp(-s - bias));
  if(derivative) value *= 1.0 - value;
  return value;
}

ntype LSTM_g::gain(int j, int i)
{
  if(gater.find(pkey(j,i)) != gater.end())
    return activation[gater[pkey(j,i)]];

  if(weight.find(pkey(j,i)) != weight.end())
    return 1.0;

  return 0.0;
}

ntype LSTM_g::theTerm(int j, int k)
{
  ntype term = 0.0;
  if(gater.find(pkey(k,k)) != gater.end() && j == gater[pkey(k,k)])
    term = oldState[k];
  for(pmap::iterator it=gater.begin(); it != gater.end(); it++) {
    pkey key = it->first;
    if(key.first == k && key.second != k && j == gater[pkey(k, key.second)])
      term += weight[pkey(k, key.second)] * oldActivation[k][key.second];
  }
  return term;
}

void LSTM_g::build(vector<vector<string> >& specData)
{
  //state.clear();
  activation.clear();
  weight.clear();
  gater.clear();
  //trace.clear();
  //extendedTrace.clear();

  numInputs = atoi(specData[0][0].c_str());
  numOutputs = atoi(specData[0][1].c_str());

  bool newNetwork = true;

  for(int line = 1; line<specData.size(); line++) {
    vector<string>& args = specData[line];
    int a0 = atoi(args[0].c_str()); 
    if(args.size() < 3) {
      newNetwork = false;
      state[a0] = atof(args[1].c_str());
      activation[a0] = actFunc(state[a0], false);
    }
    int a1 = atoi(args[1].c_str());
    if(newNetwork) {
      weight[pkey(a0, a1)] = atof(args[2].c_str());
      if(args[3] != "-1") {
	gater[pkey(a0, a1)] = atoi(args[3].c_str());
      }
      if(a0 != a1) {
	trace[pkey(a0, a1)] = 0.0;
      }
    }
    else if(args.size() > 3) {
      extendedTrace[tkey(a0, a1, atoi(args[2].c_str()))] = atof(args[3].c_str());
    }
    else if(args.size() > 2) {
      trace[pkey(a0, a1)] = atof(args[2].c_str());
    }
  }
  if(newNetwork) {
    for(nmap::iterator it = trace.begin(); it != trace.end(); ++it) {
      pkey key = it->first;
      state[key.first] = 0;
      activation[key.first] = 0;
      for(pmap::iterator git = gater.begin(); git != gater.end(); ++git) {
	pkey gkey = git->first;
	if(key.first < gkey.first && key.first == gater[gkey]) {
	  extendedTrace[tkey(key.first, key.second, gkey.first)] = 0;
	}
      }
    }
  }
  numUnits = state.size();
}

void LSTM_g::toLowLevel(vector<vector<string> >& specData)
{
  numInputs = atoi(specData[0][0].c_str());
  numOutputs = atoi(specData[0][1].c_str());
  inputToOutput = atoi(specData[0][2].c_str());
  biasOutput = atoi(specData[0][3].c_str());

  int lastInput = numInputs - biasOutput;
  vector<vector<string> > blockData;
  vector<vector<int> > connections;

  int maxBlockData = 0;
  for(int line = 1; line<specData.size(); line++) {
    vector<string>& args = specData[line];
    if(args.size() > 3) {
      int bd = atoi(args[0].c_str());
      if(maxBlockData < bd) maxBlockData = bd;
      blockData.push_back(args);
      if(args[3] == "1")
	lastInput = numInputs - 1;
    }
    else if(args.size() > 2) {
      vector<int> cl;
      cl.push_back(atoi(args[0].c_str()));
      cl.push_back(atoi(args[1].c_str()));
      cl.push_back(atoi(args[2].c_str()));
      connections.push_back(cl);
    }
    else {
      layerData[atoi(args[0].c_str())] = atoi(args[1].c_str());
    }
  }
  numUnits = numInputs + 4 * maxBlockData + 4 + numOutputs;
  ddddd
}

void LSTM_g::addConnection(vector<vector<string> >& specData, int j, int i, int g)
{
  vector<string> line;
  stringstream ss;
  std::default_random_engine generator;
  std::uniform_real_distribution<ntype> distribution(-0.1, 0.1);
  ss << j;
  line.push_back(ss.str());
  ss.str("");
  ss << i;
  line.push_back(ss.str());
  ss.str("");
  ss << distribution(generator);
  line.push_back(ss.str());
  ss.str("");
  ss << g;
  line.push_back(ss.str());
  specData.push_back(line);
}

vector<int> LSTM_g::unitsInBlock(int blockNum)
{
  int start = numInputs + 4 * blockNum;
  int end = start + 4;
  int step = 4;
  for(auto it = layerData.begin(); it != layerData.end(); ++it) {
    int firstBlockInLayer = it->first;
    int layerSize = it->second;
    if((blockNum-firstBlockInLayer) >= 0 && (blockNum-firstBlockInLayer) < layerSize) {
      int offset = numInputs + 3 * firstBlockInLayer + blockNum;
      start = offset;
      end = offset + 4 * layerSize;
      step = layerSize;
      break;
    }
  }
  vector<int> result;
  for(int i=start; i<end; i+= step) {
    result.push_back(i);
  }
  return result;
}

