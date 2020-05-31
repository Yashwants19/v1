.ONESHELL:
.PHONY: test deps download build clean docker

# Go version to use when building Docker image
GOVERSION?=1.13.1

# Temporary directory to put files into.
TMP_DIR?=/tmp/

# Package list for each well-known Linux distribution
RPMS = cmake curl git unzip boost-devel boost-test boost-program-options         \
       boost-math armadillo-devel
DEBS = unzip build-essential cmake curl git pkg-config libboost-math-dev         \
       libboost-program-options-dev libboost-test-dev libboost-serialization-dev \
       libarmadillo-dev

# Detect Linux distribution
distro_deps=
ifneq ($(shell which dnf 2>/dev/null),)
	distro_deps=deps_fedora
else
ifneq ($(shell which apt-get 2>/dev/null),)
	distro_deps=deps_debian
else
ifneq ($(shell which yum 2>/dev/null),)
	distro_deps=deps_rh_centos
endif
endif
endif

# Install all necessary dependencies.
deps: $(distro_deps)
	rm -rf $(TMP_DIR)arma
	mkdir $(TMP_DIR)arma
	cd $(TMP_DIR)arma
	curl -Lo arma.tar.xz https://ftp.fau.de/macports/distfiles/armadillo/armadillo-8.400.0.tar.xz
	tar -xvJ arma.tar.xz && rm arma.tar.xz && cd -
	cd $(TMP_DIR)arma/armadillo-8.400.0
	cmake . && make && sudo make install && cd -
	rm -rf  $(TMP_DIR)arma

deps_rh_centos:
	sudo yum -y install pkgconfig $(RPMS)

deps_fedora:
	sudo dnf -y install pkgconf-pkg-config $(RPMS)

deps_debian:
	sudo apt-get -y update
	sudo apt-get -y install $(DEBS)

# Download mlpack source.
download:
	rm -rf $(TMP_DIR)mlpack
	mkdir $(TMP_DIR)mlpack
	cd $(TMP_DIR)mlpack
	curl -Lo mlpack.zip https://codeload.github.com/Yashwants19/mlpack/zip/go-bindings
	unzip -q mlpack.zip
	rm mlpack.zip
	cd -

# Build mlpack(go shared libraries).
build:
	cd $(TMP_DIR)mlpack/mlpack-go-bindings
	mkdir build
	cd build
	cmake -D BUILD_TESTS=OFF           \
	      -D BUILD_JULIA_BINDINGS=OFF  \
	      -D BUILD_PYTHON_BINDINGS=OFF \
	      -D BUILD_CLI_EXECUTABLES=OFF \
	      -D BUILD_GO_BINDINGS=OFF     \
	      -D BUILD_GO_SHLIB=ON  ..
	$(MAKE) -j $(shell nproc --all)
	$(MAKE) preinstall
	cd -

# Cleanup temporary build files.
clean:
	go clean --cache
	rm -rf $(TMP_DIR)mlpack

# Do everything.
install: deps download build sudo_install clean test


# Install system wide.
sudo_install:
	cd $(TMP_DIR)mlpack/mlpack-go-bindings/build
	sudo $(MAKE) install
	sudo ldconfig
	cd -
# Runs tests.
test:
	go test -v . ./tests

docker:
	docker build --build-arg GOVERSION=$(GOVERSION) .

