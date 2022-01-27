#!/PATH/to/python
# The main summarising script for the NccPy package

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)

import sys
import os
from NccPy_main_aux import para_custom, para_light, para_autoextend, help_display, package_content_display
from NccPy_GS_Ce import nccpy_gs_ce
from NccPy_GS_Hy import nccpy_gs_hy
from NccPy_Inv_HyCe import nccpy_inv_hyce
from NccPy_Inv_solo import nccpy_inv_solo
from NccPy_Inv_HyCe_BS import nccpy_inv_hyce_bs
from NccPy_plot_map import nccpy_plot_map


# Get the parameters from the input files
def initialize():
    print(f'===== Initializing and run the NccPy package (V0.6.0) =====\n')

    # Check if parameters/ files are specified
    # Initializing mode: nothing is specified
    if len(sys.argv) == 1:

        # Display description for help/ introductory mode
        help_display()

        # Output simplified parameter file
        para_light()

    else:
        # Custom parameter mode: output a template input file for parameter modification
        if sys.argv[1] == f'-para_custom':

            # If catalog is provided: update the output template to best-fitting grid sizes, etc
            if len(sys.argv) > 2:
                cat_in = sys.argv[2]
            else:
                cat_in = []

            # Create master edit-able parameter list file
            para_custom(cat_in)

        # Standard run mode --> generate the detailed parameter files if necessary
        elif sys.argv[1] == f'-run_all':

            # If the parameter file is not provided
            if len(sys.argv) == 2:
                print(f'<Error>  {sys.argv[1]} mode: please provide a parameter file\n')

            # If the number of inputs appear to be sufficient
            elif len(sys.argv) > 2:

                # Make sure the parameter file exists and is readable first
                if os.access(sys.argv[2], os.R_OK):

                    # Append complete parameter file from the simplified one if necessary
                    para_file_pass = para_autoextend(sys.argv[2])

                    # Run the lot if no error is indicated above
                    if para_file_pass:
                        print(f'>>>> Run {sys.argv[1]} mode with parameters in: {para_file_pass} <<<<')
                        run_all(para_file_pass)
                        print(f'===== Finished run: {sys.argv[1]} mode')

                # Parameter file not readable
                else:
                    print(f'<Error> The parameter file is not readable: {sys.argv[2]}\n')

        # Custom run mode --> run specified programs only
        elif sys.argv[1] == f'-run_single':

            # If either/ both the program is not specified and the parameter file is not provided
            if len(sys.argv) == 2 or len(sys.argv) == 3:
                print(f'<Info> {sys.argv[1]} mode: please specify the program to run AND provide parameter file\n')

                # Display the package contents
                package_content_display()

            # If the number of inputs appear to be sufficient
            else:

                # Make sure the parameter file exists and is readable first
                if os.access(sys.argv[3], os.R_OK):

                    print(f'>>>> Run {sys.argv[1]} mode with parameters in: {sys.argv[3]} <<<<')

                    # Run the specified program
                    if sys.argv[2] == f'nccpy_gs_hy':
                        nccpy_gs_hy(sys.argv[3])
                    elif sys.argv[2] == f'nccpy_gs_ce':
                        nccpy_gs_ce(sys.argv[3])
                    elif sys.argv[2] == f'nccpy_inv_hyce':
                        nccpy_inv_hyce(sys.argv[3])
                    elif sys.argv[2] == f'nccpy_inv_solo':
                        nccpy_inv_solo(sys.argv[3])
                    elif sys.argv[2] == f'nccpy_inv_hyce_bs':
                        nccpy_inv_hyce_bs(sys.argv[3])
                    elif sys.argv[2] == f'nccpy_plot_map':
                        nccpy_plot_map(sys.argv[3])

                    # Unknown function!
                    else:
                        print(f'<Error> In {sys.argv[1]} mode: unknown function {sys.argv[2]}\n')

                # Parameter file not readable
                else:
                    print(f'<Error> The parameter file is not readable: {sys.argv[3]}\n')

    return None


# Script to run through the whole set of programs
def run_all(para_file_pass):

    nccpy_gs_hy(para_file_pass)
    nccpy_gs_ce(para_file_pass)
    nccpy_inv_hyce(para_file_pass)
    nccpy_inv_solo(para_file_pass)
    nccpy_plot_map(para_file_pass)
    # nccpy_inv_hyce_bs(para_file_pass)

    return None


initialize()
