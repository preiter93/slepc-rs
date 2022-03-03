hello: hello.o
	-${CLINKER} -o hello hello.o ${SLEPC_SYS_LIB}
	${RM} hello.o

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex1: ex1.o
	-${CLINKER} -o ex1 ex1.o ${SLEPC_EPS_LIB}
	${RM} ex1.o
