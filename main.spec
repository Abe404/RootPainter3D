#Spec file describing how to build executable for Ubuntu and Windows

#Copyright (C) 2022 André Pedersen

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.

# -*- mode: python ; coding: utf-8 -*-

# necessary for MacOS
import os
os.environ['LC_CTYPE'] = "en_US.UTF-8"
os.environ['LANG'] = "en_US.UTF-8"

from numpy import loadtxt
import shutil

block_cipher = None

# fix hidden imports
hidden_imports = loadtxt("./painter/requirements.txt", comments="#", delimiter=",", unpack=False, dtype=str)

for i in ['pywt._extensions._cwt', 'skimage.io._plugins.pil_plugin', 'pkg_resources.py2_warn']:
    hidden_imports.append(i)

hidden_imports = [x.lower() for x in hidden_imports]

# copy dependencies - to get icons
shutil.copytree("./painter/", "./tmp_dependencies/painter/")

a = Analysis(['./painter/src/main.py'],
             pathex=['.'],
             binaries=[],
             datas=[],
             hiddenimports=hidden_imports,
             hookspath=["./painter/hooks/"],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher
)

# to compile everything into a macOS Bundle (.APP)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='RootPainter3D',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon="./tmp_dependencies/painter/icons/Icon.ico"
)
coll = COLLECT(exe,
               a.binaries,
               Tree("./tmp_dependencies/"),
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='RootPainter3D'
)
