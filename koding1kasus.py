import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime

def energiawal(grid):
    N = grid.shape[0]
    total_energi = 0
    for i in range(N): 
        for j in range(N): 
            S = grid[i,j]
            s_neighbors = (
                grid[(i+1)%N, j] + grid[(i-1)%N, j] +
                grid[i, (j+1)%N] + grid[i, (j-1)%N]
            )
            total_energi += -S*s_neighbors
    return total_energi/2

def metropolis_step(grid, T):
    N = grid.shape[0]
    x, y = random.randint(0, N-1), random.randint(0, N-1)
    s_neighbors = (
        grid[(x+1)%N, y] + grid[(x-1)%N, y] +
        grid[x, (y+1)%N] + grid[x, (y-1)%N]
    )
    dE_acceptance = 0
    delta_E = 2 * grid[x, y] * s_neighbors
    if delta_E < 0 or random.random() < np.exp(-delta_E /T):
        grid[x, y] *= -1
        dE_acceptance = delta_E
    return grid, dE_acceptance

def run_simulation(N=20, temp=1.0, n_steps=100000):
    grid = np.random.choice([-1, 1], size=(N, N))
    magnetization_history = []
    Elok_history = []
    energisistem = energiawal(grid)
    for step in range(n_steps):
        grid, dE_accept = metropolis_step(grid, temp)
        energisistem += dE_accept

        if step % 100 == 0:
            magnetization = np.mean(grid)
            magnetization_history.append(magnetization)
            Elok_history.append(energisistem/(N**2))

    return grid, magnetization_history, Elok_history

def plot(grid_size, temperatures, monte_carlo_steps, run_index=1):
    waktu_sekarang = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    fig.suptitle('2D Ising Model Simulation via Metropolis Algorithm')
    final_grid, M_history, E_history = run_simulation(N=grid_size, temp=temperatures, n_steps=monte_carlo_steps)

    ax_grid = axes[0]
    ax_grid.imshow(final_grid, cmap='binary', vmin=-1, vmax=1)
    ax_grid.set_title(f"Final State at T = {temperatures:.2f}")

    ax_mag = axes[1]
    ax_mag.plot(M_history)
    ax_mag.set_title(f"Magnetization at T = {temperatures:.2f}")
    ax_mag.set_xlabel("Monte Carlo Steps (x100)")
    ax_mag.set_ylabel("Average Magnetization")
    ax_mag.set_ylim(-1.1, 1.1)

    ax_e = axes[2]
    ax_e.plot(E_history)
    ax_e.set_title(f"Energy at T = {temperatures:.2f}")
    ax_e.set_xlabel("Monte Carlo Steps (x100)")
    ax_e.set_ylabel("Average local Energy")
    teks_legenda = mpatches.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='none')
    sus = np.var((np.array(M_history)))/temperatures
    kap = np.var((np.array(E_history)))/(temperatures**2)

    plt.legend([teks_legenda], [f'Rata-rata |M| =  {abs(np.mean(M_history)):.6f}\n Rata-rata E_lok = {abs(np.mean(E_history)):.6f}\n E_lok last state = {E_history[-1]:.6f} \n x = {sus}, C = {kap}'], handlelength=0)
    plt.tight_layout()
    
    # Otomatis buat folder 'hasil' jika belum ada
    output_dir = "hasil"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Simpan hasil plot otomatis ke dalam folder hasil
    filename = f"{output_dir}/G{grid_size}T{temperatures}R{run_index}_{waktu_sekarang}.png"
    plt.savefig(filename)
    plt.show()
    plt.close(fig) # Menutup figure agar hemat memori
    return M_history, E_history

# --- FUNGSI LOGIKA GUI ---
def jalankan_simulasi():
    try:
        # Ambil nilai dari input GUI
        grid_size = int(entry_grid.get())
        temperatures = float(entry_temp.get())
        monte_carlo_steps = int(entry_steps.get())
        
        opsi_iterasi = var_opsi.get()
        
        if opsi_iterasi == "1":
            plot(grid_size, temperatures, monte_carlo_steps, run_index=1)
            messagebox.showinfo("Sukses", "Simulasi selesai! Plot disimpan di folder 'hasil'.")
        else:
            jumlah_iterasi = int(entry_iterasi.get())
            for i in range(jumlah_iterasi):
                plot(grid_size, temperatures, monte_carlo_steps, run_index=i+1)
            messagebox.showinfo("Sukses", f"Simulasi sebanyak {jumlah_iterasi} kali selesai! Semua plot disimpan di folder 'hasil'.")
            
    except ValueError:
        messagebox.showerror("Error", "Pastikan semua input diisi dengan angka yang benar!")

def toggle_input_iterasi():
    # Mengaktifkan/menonaktifkan input jumlah iterasi berdasarkan pilihan radio button
    if var_opsi.get() == "2":
        entry_iterasi.config(state="normal")
    else:
        entry_iterasi.config(state="disabled")

# --- DESAIN INTERFACES GUI ---
root = tk.Tk()
root.title("Simulasi 2D Ising Model")
root.geometry("400x380")
root.resizable(False, False)

# Mengatur bobot kolom di root agar frame bisa melebar jika window membesar
root.columnconfigure(0, weight=1)

# Frame Input Utama
frame_input = ttk.LabelFrame(root, text=" Parameter Simulasi ", padding=15)
frame_input.pack(fill="x", padx=15, pady=10)

# Mengatur bobot kolom di dalam frame_input agar kolom entry (kolom 1) bisa melebar
frame_input.columnconfigure(1, weight=1)

# Input Grid Size
ttk.Label(frame_input, text="Grid Size:").grid(row=0, column=0, sticky="w", pady=5)
entry_grid = ttk.Entry(frame_input)
entry_grid.insert(0, "20") # nilai default
# PERBAIKAN: Mengganti fill dan expand dengan sticky="ew"
entry_grid.grid(row=0, column=1, sticky="ew", pady=5)

# Input Temperature
ttk.Label(frame_input, text="Temperature (T):").grid(row=1, column=0, sticky="w", pady=5)
entry_temp = ttk.Entry(frame_input)
entry_temp.insert(0, "1.0")
# PERBAIKAN: Mengganti fill dan expand dengan sticky="ew"
entry_temp.grid(row=1, column=1, sticky="ew", pady=5)

# Input Monte Carlo Steps
ttk.Label(frame_input, text="Monte Carlo Steps:").grid(row=2, column=0, sticky="w", pady=5)
entry_steps = ttk.Entry(frame_input)
entry_steps.insert(0, "100000")
# PERBAIKAN: Mengganti fill dan expand dengan sticky="ew"
entry_steps.grid(row=2, column=1, sticky="ew", pady=5)

# Frame Pilihan Iterasi
frame_opsi = ttk.LabelFrame(root, text=" Opsi Iterasi ", padding=15)
frame_opsi.pack(fill="x", padx=15, pady=5)

var_opsi = tk.StringVar(value="1")

r1 = ttk.Radiobutton(frame_opsi, text="Jalankan 1 Kali Semisal", value="1", variable=var_opsi, command=toggle_input_iterasi)
r1.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

r2 = ttk.Radiobutton(frame_opsi, text="Ulangi dengan jumlah iterasi:", value="2", variable=var_opsi, command=toggle_input_iterasi)
r2.grid(row=1, column=0, sticky="w", pady=2)

entry_iterasi = ttk.Entry(frame_opsi, width=10, state="disabled")
entry_iterasi.insert(0, "2")
entry_iterasi.grid(row=1, column=1, sticky="w", padx=5, pady=2)

# Tombol Run
btn_run = ttk.Button(root, text="MULAI SIMULASI", command=jalankan_simulasi)
btn_run.pack(fill="x", padx=15, pady=20)

root.mainloop()
