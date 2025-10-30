from mpi4py import MPI
import numpy as np


def arranque():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #print(f"Soy el proceso {rank} de un total de {size} procesos")
    return comm, rank, size





class Punto3D:
    def __init__(self, x=0.0, y=0.0, z=0.0 , id=0):
        self.x = x
        self.y = y
        self.z = z
        self.id = id


    def __str__(self):
        return f"Punto(id={self.id}) x=({self.x}, y={self.y}, z={self.z})"



def main_1():
    comm, rank, size = arranque()
    
    punto_dtype = np.dtype([
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
        ('id', np.int32)
    ],  align=True) # Forzar alineación

    
    # # 3 Doubles + 1 Int ( pero usamos 4 doubles para alinear )
    # tipo_punto = MPI.DOUBLE.Create_contiguous(4)

    # # 4 elementos double para cubrir 3 doubles + 1 int
    # tipo_punto = MPI.DOUBLE.Create_contiguous(4)


    # bloques = [ 1 , 1, 1, 1 ]
    # tipos = [ MPI.DOUBLE, MPI.DOUBLE, MPI.DOUBLE, MPI.INT ]


    # ej_array = [1.0, 2.0, 3.0, 4.0]
    # buffer_ejemplo = np.array(ej_array, dtype=punto_dtype)
    
    
    # # Obtener offsets usando la estructura del dtype
    # desplazamientos = [
    #     punto_dtype.fields['x'][1],
    #     punto_dtype.fields['y'][1],
    #     punto_dtype.fields['z'][1],
    #     punto_dtype.fields['id'][1]
    # ]
    
    
    # tamaño_total = punto_dtype.itemsize
    
    
    # if tamaño_total % MPI.DOUBLE.Get_size() != 0:
    #     # Añadir padding para alinear
    #     padding = MPI.DOUBLE.Get_size() - (tamaño_total % MPI.DOUBLE.Get_size())
    #     bloques.append(padding // MPI.DOUBLE.Get_size())
    #     desplazamientos.append(tamaño_total)


    # tipo_punto = MPI.Datatype.Create_struct(bloques, desplazamientos, tipos)
    # tipo_punto.Commit()


    if rank == 0:
        # Proceso 0 crea y envía el punto
        
        punto_data = np.array([(1.0, 2.0, 3.0, 42)], dtype=punto_dtype)
        print(f"Proceso {rank} envió el punto x: {punto_data['x'][0]}, y: {punto_data['y'][0]}, z: {punto_data['z'][0]}, id: {punto_data['id'][0]}")
        # Usar Send con el conteo explícito

        #contexto = [punto_data, tipo_punto]
        contexto = [punto_data, MPI.BYTE]
        comm.Send(contexto, dest=1, tag=1)
        # punto_obj = Punto3D(1.5,2.5,3.5,100)
        # print(f"Proceso {rank} envió: {punto_obj}")
        # print(f"Tamaño del buffer {punto_data.nbytes} bytes")
        # print(f"Extent del tipo: {tipo_punto.Get_extent()[1]} bytes")

    elif rank == 1:
        # Proceso 1 recibe los datos y reconstruye el objeto
        punto_data = np.empty(1, dtype=punto_dtype)
        # comm.Recv([punto_data, tipo_punto], source=0, tag=1)
        comm.Recv([punto_data, MPI.BYTE], source=0, tag=1)
        
        # punto_recibido = Punto3D(
        #     punto_data["x"][0],
        #     punto_data["y"][0],
        #     punto_data["z"][0],
        #     punto_data["id"][0]
        # )
        print(f"Proceso {rank} recibe el punto: x: {punto_data['x'][0]}, y: {punto_data['y'][0]}, z: {punto_data['z'][0]}, id: {punto_data['id'][0]}")
    
    # Limpiar el tipo de dato personalizado
    #tipo_punto.Free()



if __name__ == "__main__":
    main_1()