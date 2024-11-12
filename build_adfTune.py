import PyInstaller.__main__, os, shutil

PyInstaller.__main__.run([
    'optimization.py',
    '--clean',
    '--log-level=WARN',
    '--noconfirm',
    '--onefile',
    '--console',
    '--distpath=./',
    '-n=Thy_ADFtune',
])

# remove the build directory
shutil.rmtree('build')
# remove the spec file
os.remove('./Thy_ADFtune.spec')