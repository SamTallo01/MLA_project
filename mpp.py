import os
import csv
import openslide

def estrai_mpp_da_cartella(cartella, output_csv="mpp_ndpi.csv"):
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_ndpi", "mpp_x", "mpp_y", "mpp_media"])

        for file in os.listdir(cartella):
            if file.lower().endswith(".ndpi"):
                percorso = os.path.join(cartella, file)

                try:
                    slide = openslide.OpenSlide(percorso)

                    mpp_x = slide.properties.get("openslide.mpp-x")
                    mpp_y = slide.properties.get("openslide.mpp-y")

                    if mpp_x and mpp_y:
                        mpp_x = float(mpp_x)
                        mpp_y = float(mpp_y)
                        mpp_media = (mpp_x + mpp_y) / 2
                    else:
                        mpp_media = None

                    writer.writerow([file, mpp_x, mpp_y, mpp_media])
                    slide.close()

                except Exception as e:
                    print(f"Errore con {file}: {e}")
                    writer.writerow([file, None, None, None])

    print(f"CSV salvato in: {output_csv}")


if __name__ == "__main__":
    cartella_ndpi = "data"
    estrai_mpp_da_cartella(cartella_ndpi)
