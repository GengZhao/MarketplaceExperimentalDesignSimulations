CXX=g++
OPTFLAG=-O3
CXXFLAGS=-Wall -g $(OPTFLAG) -std=c++14

EXECUTABLES=main eval_estimators varying_design lrcr_vary_design conv_tradeoff vary_treat

all: $(EXECUTABLES)

main: main.o Marketplace.o
	$(CXX) $(CXXFLAGS) -o main main.o Marketplace.o

conv_tradeoff: conv_tradeoff.o Marketplace.o
	$(CXX) $(CXXFLAGS) -pthread -o conv_tradeoff conv_tradeoff.o Marketplace.o

eval_estimators: eval_estimators.o Marketplace.o
	$(CXX) $(CXXFLAGS) -pthread -o eval_estimators eval_estimators.o Marketplace.o

lrcr_vary_design: lrcr_vary_design.o Marketplace.o
	$(CXX) $(CXXFLAGS) -pthread -o lrcr_vary_design lrcr_vary_design.o Marketplace.o

vary_treat: vary_treat.o Marketplace.o
	$(CXX) $(CXXFLAGS) -pthread -o vary_treat vary_treat.o Marketplace.o

varying_design: varying_design.o Marketplace.o
	$(CXX) $(CXXFLAGS) -pthread -o varying_design varying_design.o Marketplace.o

main.o: main.cc Marketplace.h
	$(CXX) $(CXXFLAGS) -c main.cc

conv_tradeoff.o: conv_tradeoff.cc Marketplace.h utils.h
	$(CXX) $(CXXFLAGS) -c conv_tradeoff.cc

eval_estimators.o: eval_estimators.cc Marketplace.h utils.h
	$(CXX) $(CXXFLAGS) -c eval_estimators.cc

lrcr_vary_design.o: lrcr_vary_design.cc Marketplace.h utils.h
	$(CXX) $(CXXFLAGS) -c lrcr_vary_design.cc

vary_treat.o: vary_treat.cc Marketplace.h utils.h
	$(CXX) $(CXXFLAGS) -c vary_treat.cc

varying_design.o: varying_design.cc Marketplace.h utils.h
	$(CXX) $(CXXFLAGS) -c varying_design.cc

Marketplace.o: Marketplace.cc Marketplace.h utils.h
	$(CXX) $(CXXFLAGS) -c Marketplace.cc

clean:
	$(RM) $(EXECUTABLES) *.o *~

