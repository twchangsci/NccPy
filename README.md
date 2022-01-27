# NccPy
An earthquake relocation package to improve hypocenter and centroid locations using the network correlation coefficient (NCC) method

## Quick start: four steps toward your first results:

### I. Download, unpack, and store the package
### II. Environment setup:
A. Set the path to the package (e.g. for Unix and Linux users:)
```
export PATH="$PATH:(path to package)"
```
B. Modify the first line of `NccPy_main.py` to the path of your Python executable.

C. (For Mac users) To avoid bugs, navigate to the `Source` directory, and:
```
xattr -d com.apple.quarantine NccPy_main.py
```

### III. Warm up exercise: execute demo
Navigate to the `Demo` directory, and:
```
NccPy_main.py -run_all para_input_light.txt
```  

### IV. Now try it with your own datasets!
Navigate to your working directory, and:
```
NccPy_main.py
```
Everything you need will be displayed on the splash screen/ prepared for you.

For full instructions, kindly refer to the manual (Manual.pdf).
