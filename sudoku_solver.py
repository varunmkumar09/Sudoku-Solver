from keras.models import load_model
import numpy as np
import cv2

model = load_model('digit_recognition_model_v2.h5')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def find_sudoku_grid(image):
    contours, _ = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sudoku_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            sudoku_contour = approx
            max_area = area
    return sudoku_contour

def extract_cells(image, sudoku_contour):
    x, y, w, h = cv2.boundingRect(sudoku_contour)
    sudoku_grid_region = image[y:y + h, x:x + w]
    cell_size = h // 9
    cells = [sudoku_grid_region[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size]
             for row in range(9) for col in range(9)]
    return cells

def recognize_digits(cell_images):
    digits = []
    for cell_image in cell_images:
        cell_image_cropped = cell_image[6:-6, 6:-6]
        cell_image_resized = cv2.resize(cell_image_cropped, (28, 28))
        cell_image_normalized = cell_image_resized / 255.0
        cell_image_normalized[cell_image_normalized > 0.2] = 1
        cell_image_final = np.expand_dims(cell_image_normalized, axis=0)
        cell_image_column = np.array(cell_image_final)
        col_counts = np.sum(cell_image_column[:, 6:25] > 0.25, axis=0)
        if np.sum(col_counts) >= 3:
            cell_image_final = np.expand_dims(cell_image_final, axis=-1)
            digit = np.argmax(model.predict(cell_image_final), axis=-1)[0]
            digits.append(digit if digit != 0 else 0)
        else:
            digits.append(0)
    return digits


def solve_sudoku(grid):
    empty_cell = find_empty_cell(grid)
    if not empty_cell:
        return True
    row, col = empty_cell
    for num in range(1, 10):
        if is_valid_move(grid, row, col, num):
            grid[row][col] = num
            if solve_sudoku(grid):
                return True
            grid[row][col] = 0
    return False

def is_valid_move(grid, row, col, num):
    if num in grid[row]:
        return False
    if num in [grid[i][col] for i in range(9)]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if grid[i][j] == num:
                return False
    return True

def find_empty_cell(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return (i, j)
    return None

image = cv2.imread('') # Add the path to your unsolved sudoku image here

preprocessed_image = preprocess_image(image)

sudoku_contour = find_sudoku_grid(preprocessed_image)

if sudoku_contour is not None:
    cells = extract_cells(preprocessed_image, sudoku_contour)
    digits = recognize_digits(cells)
    sudoku_grid = np.array(digits).reshape(9, 9)
    print(sudoku_grid)
    if solve_sudoku(sudoku_grid):
        print("Sudoku solved successfully:")
        print(sudoku_grid)
    else:
        print("Failed to solve Sudoku.")
else:
    print("Sudoku grid not found in the image.")
