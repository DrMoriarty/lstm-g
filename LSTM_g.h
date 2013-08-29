#include <string>
#include <queue>
#include <vector>
#include <math.h>

typedef double ntype;

class LSTM_g {

  LSTM_g();
  
  ntype actFunc(ntype s, bool derivative, ntype bias=0.0);
  ntype gain(int j, int i);
  ntype theTerm(int j, int k);
  LSTM_g* build(std::string& specData);
  void* toLowLevel(std::string& specData);
  std::string toString(bool newNetwork, char newLine='\n');
  

};
