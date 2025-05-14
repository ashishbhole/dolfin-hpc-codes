CFLAGS  = `pkg-config --cflags dolfin` -g
LIBS    = `pkg-config --libs dolfin`
CXX     = `pkg-config --variable=compiler dolfin`

DEST    = demo
OBJECTS = main.o

DIR=/cfs/klemming/projects/snic/heartsolver/joel/cylinder6

all: $(DEST)

install:
	cp demo ${DIR}
	cp submit.sh ${DIR}
	cp cylinder-gmsh.bin ${DIR}
	cp cylinder_3d_bmk_refined1.bin ${DIR}
	cp cylinder_3d_bmk.bin ${DIR}
	cp cylinder-ashish.bin ${DIR}

run:
	(cd ${DIR} && sbatch submit.sh)

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST) *.log *.vtu *.pvtu *.pvd solution* *.out

$(DEST): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $<
