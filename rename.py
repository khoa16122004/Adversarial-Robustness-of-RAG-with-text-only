import os
import sys

def rename_files(folder_path, model_name):
    if not os.path.isdir(folder_path):
        print(f"Đường dẫn không hợp lệ: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            new_filename = f"{model_name}_{filename}"
            new_path = os.path.join(folder_path, new_filename)

            # Đổi tên file
            os.rename(file_path, new_path)
            print(f"Đã đổi: {filename} → {new_filename}")

if __name__ == "__main__":


    folder_path = "llama_7b_nsgaii_logs"
    model_name = "llama-7b"

    rename_files(folder_path, model_name)