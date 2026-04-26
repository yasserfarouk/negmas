package genius.core;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 * This is a utility class to handle writing and 
 * reading {@link Serializable} objects into/from a file.
 * @author Samanta H.
 *
 */
public class SerializeHandling{

	/**
	 * Writes the data into disc, in a new file under the path given.
	 * @param data the object to be saved into a new file.
	 * @param path the final path (including the name of the file which will
	 * 			   be created).
	 * @return true if writing the object into the path was successful.
	 * 		   false otherwise.
	 */
	public static boolean writeToDisc(Serializable data, String path){
		try{

			FileOutputStream fout = new FileOutputStream(path);
			ObjectOutputStream oos = new ObjectOutputStream(fout);   
			oos.writeObject(data);
			oos.close();
//			System.out.println("Done writing.");

		}catch(Exception ex){
			ex.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * Reads an Object from the "path" in the disc.
	 * @param path the path where the object is saved in the disc.
	 * @return the object saved in path.
	 */
	public static Serializable readFromDisc(String path) {
		Serializable data;
		try{

			FileInputStream fin = new FileInputStream(path);
			ObjectInputStream ois = new ObjectInputStream(fin);
			data = (Serializable) ois.readObject();
			ois.close();
//			System.out.println("Done reading.");
			return data;

		}catch(FileNotFoundException e){
			return null;
		}catch(Exception ex){
			ex.printStackTrace();
			return null;
		} 
	}
}
