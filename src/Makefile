all:
	g++ -static -O3 -std=c++11 *.h *.cpp -I../include -pthread -o matrix_factorization

test:
	./matrix_factorization ../conf/config.json

clean:
	rm -rf ./matrix_factorization ../data/o_*
