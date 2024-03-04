Sudoku Solver

This Sudoku Solver is a Python-based application that utilizes image processing and digit recognition techniques to solve Sudoku puzzles. The application takes an image of a Sudoku puzzle as input and uses convolutional neural networks (CNNs) for digit recognition, ultimately solving the puzzle and providing the solution.

Features
Image Input: Accepts an image of a Sudoku puzzle as input.
Digit Recognition: Utilizes CNNs for recognizing digits in the Sudoku grid.
Solver Algorithm: Employs a backtracking algorithm to solve the Sudoku puzzle.

Run the main script:
1. python image_reader.py
2. After running the above code, A model will be created which will be used in the sudoku_solver.py file (The model will have around 99.2% accuracy)
3. python sudoku_solver.py
4. Provide an image of the Sudoku puzzle as input.
5. View the solved Sudoku puzzle.

Include examples of input images and corresponding solved puzzles.

Contributors
Varun Kumar M (GitHub)

Acknowledgments
This project was inspired by the desire to apply image processing and machine learning techniques to solve Sudoku puzzles.
Special thanks to the developers of OpenCV, Keras, and TensorFlow for providing powerful libraries for image processing and machine learning.
