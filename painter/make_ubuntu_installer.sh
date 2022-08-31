mkdir -p ubuntu_setup/usr/local/bin
cp -r target/RootPainter3D ubuntu_setup/usr/local/bin
dpkg-deb --build --root-owner-group ubuntu_setup
cp -r ubuntu_setup.deb target/RootPainter3D-ubuntu.deb
