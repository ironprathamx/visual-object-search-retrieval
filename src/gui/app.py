import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieve import VisualSearchEngine


class VisualSearchGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Visual Object Search and Retrieval")
        self.root.geometry("1000x800")

        self.query_path = None
        self.query_photo = None

        # Load retrieval engine once
        self.engine = VisualSearchEngine()

        # ---------------------------------------------
        # Title
        # ---------------------------------------------

        title = tk.Label(
            root,
            text="Visual Object Search and Retrieval",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=20)

        # ---------------------------------------------
        # Select Image
        # ---------------------------------------------

        select_button = tk.Button(
            root,
            text="Select Query Image",
            command=self.select_image,
            font=("Arial", 14),
            padx=20,
            pady=10
        )
        select_button.pack(pady=10)

        # ---------------------------------------------
        # Query Image
        # ---------------------------------------------

        self.image_label = tk.Label(
            root,
            text="No image selected",
            font=("Arial", 14)
        )
        self.image_label.pack(pady=20)

        # ---------------------------------------------
        # Search
        # ---------------------------------------------

        search_button = tk.Button(
            root,
            text="SEARCH",
            command=self.search,
            font=("Arial", 16, "bold"),
            padx=40,
            pady=12
        )
        search_button.pack(pady=20)

        # ---------------------------------------------
        # Results
        # ---------------------------------------------

        self.results_frame = tk.Frame(root)
        self.results_frame.pack(pady=20)

    def select_image(self):

        path = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        self.query_path = path

        image = Image.open(path)
        image.thumbnail((500, 400))

        self.query_photo = ImageTk.PhotoImage(image)

        self.image_label.config(
            image=self.query_photo,
            text=""
        )

    def search(self):

        if self.query_path is None:
            messagebox.showwarning(
                "No Image",
                "Please select a query image first."
            )
            return

        try:

            output = self.engine.search(
                self.query_path,
                4
            )

            # -----------------------------------------
            # Clear previous results
            # -----------------------------------------

            for widget in self.results_frame.winfo_children():
                widget.destroy()

            results = output["results"]

            # -----------------------------------------
            # Display results
            # -----------------------------------------

            for i, result in enumerate(results):

                image_path = (
                    "../dataset/ukbench/" + result["image"]
                )

                image = Image.open(image_path)

                image.thumbnail((180, 180))

                photo = ImageTk.PhotoImage(image)

                card = tk.Frame(
                    self.results_frame,
                    bd=2,
                    relief="solid",
                    padx=10,
                    pady=10
                )

                card.grid(
                    row=0,
                    column=i,
                    padx=10
                )

                # Keep reference to image
                card.photo = photo

                image_label = tk.Label(
                    card,
                    image=photo
                )

                image_label.pack()

                rank_label = tk.Label(
                    card,
                    text=f"Rank {result['rank']}",
                    font=("Arial", 12, "bold")
                )

                rank_label.pack(pady=(8, 2))

                inlier_label = tk.Label(
                    card,
                    text=f"Inliers: {result['inliers']}",
                    font=("Arial", 11)
                )

                inlier_label.pack()

                tfidf_label = tk.Label(
                    card,
                    text=f"TF-IDF: {result['tfidf_score']:.4f}",
                    font=("Arial", 10)
                )

                tfidf_label.pack()

            # -----------------------------------------
            # Processing time
            # -----------------------------------------

            time_label = tk.Label(
                self.results_frame,
                text=(
                    f"Search completed in "
                    f"{output['total_time']:.2f} seconds"
                ),
                font=("Arial", 12)
            )

            time_label.grid(
                row=1,
                column=0,
                columnspan=4,
                pady=20
            )

        except Exception as e:

            messagebox.showerror(
                "Search Error",
                str(e)
            )


# ---------------------------------------------
# Run application
# ---------------------------------------------

root = tk.Tk()

app = VisualSearchGUI(root)

root.mainloop()