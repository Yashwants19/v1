.ONESHELL:
.PHONY: test deps download build clean astyle cmds docker

# Go version to use when building Docker image
GOVERSION?=1.13.1

# Temporary directory to put files into.
TMP_DIR?=/tmp/

# Package list for each well-known Linux distribution
RPMS=cmake curl git unzip boost-devel boost-test boost-program-options boost-math armadillo-devel binutils-devel
DEBS=unzip build-essential cmake curl git pkg-config libboost-math-dev libboost-program-options-dev libboost-test-dev libboost-serialization-dev libarmadillo-dev binutils-dev

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

deps_rh_centos:
	sudo yum -y install pkgconfig $(RPMS)

deps_fedora:
	sudo dnf -y install pkgconf-pkg-config $(RPMS)

deps_debian:
	sudo apt-get -y update
	sudo apt-get -y install $(DEBS)


# Download mlpack source tarballs.
download:
	rm -rf $(TMP_DIR)mlpack
	mkdir $(TMP_DIR)mlpack
	cd $(TMP_DIR)mlpack
	curl -Lo mlpack.zip https://codeload.github.com/Yashwants19/mlpack/zip/go-bindings
	unzip -q mlpack.zip
	rm mlpack.zip
	cd -

# Build mlpack.
build:
	cd $(TMP_DIR)mlpack/mlpack-go-bindings
	mkdir build
	cd build
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_TESTS=OFF -D BUILD_JULIA_BINDINGS=OFF -D BUILD_PYTHON_BINDINGS=OFF -D BUILD_CLI_EXECUTABLES=OFF -D BUILD_GO_BINDINGS=ON ..
	$(MAKE) -j $(shell nproc --all)
	$(MAKE) preinstall
	cd -

# Cleanup temporary build files.
clean:
	go clean --cache
	rm -rf $(TMP_DIR)mlpack

# Do everything.
install: deps download build sudo_install clean verify

# Install system wide.
sudo_install:
	cd $(TMP_DIR)mlpack/mlpack-go-bindings/build
	sudo $(MAKE) install
	sudo ldconfig
	cd -

# Build a minimal Go app to confirm gocv works.
verify:
	go run main.go

# Runs tests.
# This assumes env.sh was already sourced.
# pvt is not tested here since it requires additional depenedences.
test:
	go test . ./tests
