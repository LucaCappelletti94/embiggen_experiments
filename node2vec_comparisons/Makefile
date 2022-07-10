install_snap:
	git clone https://github.com/snap-stanford/snap
	cd snap
	make -j$(nproc) all
	cd ..

install:
	make install_snap
	python3 -m pip install pip -U
	python3 -m pip install -r requirements.txt