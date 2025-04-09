CFLAGS  = `pkg-config --cflags dolfin` -g
LIBS    = `pkg-config --libs dolfin`
CXX     = `pkg-config --variable=compiler dolfin`

DEST    = demo
OBJECTS = main.o

DIR=/cfs/klemming/projects/snic/heartsolver/joel/euler3

all: $(DEST)

install:
	cp demo ${DIR}
	cp submit.sh ${DIR}
	cp long_refined.bin ${DIR}
	cp rectangle.bin ${DIR}

run:
	(cd ${DIR} && sbatch submit.sh)

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST) *.log *.vtu *.pvtu *.pvd

$(DEST): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
