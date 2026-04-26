package negotiator.boaframework.agent;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

import javax.xml.bind.DatatypeConverter;

/**
 * Series of methods to (un)serialize an object to a string to store it as a
 * file.
 * 
 * @author Tim Baarslag
 * @param <A>
 *            class of the object which is serialized.
 */
public class Serializer<A> {

	/** Path to the file in which the serialized class must be stored. */
	private final String fileName;
	/** If it should be reported if the file cannot be found. */
	private final boolean log;

	/**
	 * Create an object to serialize a class. The filename specifies the path in
	 * which the serialized class is stored. File not found exceptions are not
	 * reported.
	 * 
	 * @param fileName
	 *            path to file in which the serialized class is stored.
	 */
	public Serializer(String fileName) {
		this(fileName, false);
	}

	/**
	 * Create an object to serialize a class. The filename specifies the path in
	 * which the serialized class is stored.
	 * 
	 * @param fileName
	 *            path to file in which the serialized class is stored.
	 * @param log
	 *            specifies if file not found exceptions should be reported.
	 */
	public Serializer(String fileName, boolean log) {
		super();
		this.fileName = fileName;
		this.log = log;
	}

	/**
	 * Read a serialized object from a file and restore it.
	 * 
	 * @return unserialized object.
	 */
	@SuppressWarnings("unchecked")
	public A readFromDisk() {
		InputStream is = null;
		ObjectInputStream ois = null;
		A obj = null;

		final String errorMsg = "Error opening (" + fileName + ").\n";
		try {
			is = new BufferedInputStream(new FileInputStream(fileName), 50000 * 1024);

			ois = new ObjectInputStream(is);

			final Object readObject = ois.readObject();
			ois.close();
			is.close();
			obj = (A) readObject;
			return obj;
		} catch (FileNotFoundException e) {
			if (log)
				System.out.println(errorMsg + e);
		} catch (IOException e) {
			System.out.println(errorMsg + e);

		} catch (ClassNotFoundException e) {
			System.out.println(errorMsg + e);
		} catch (ClassCastException e) {
			System.out.println(errorMsg + e);
		}
		System.out.println(fileName + " is old; It should be rebuilt.");
		return null;
	}

	/**
	 * Serializes an object to the specified file.
	 * 
	 * @param a
	 *            object to be serialized.
	 */
	public void writeToDisk(A a) {
		OutputStream os = null;
		ObjectOutputStream oos = null;
		try {
			os = new BufferedOutputStream(new FileOutputStream(fileName));

			oos = new ObjectOutputStream(os);
			oos.writeObject(a);
			oos.close();
			os.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	/**
	 * Serializes an object to a string encoded by using Base64 to avoid
	 * characterset problems.
	 * 
	 * @param a
	 *            object to serialize.
	 * @return serialized object.
	 */
	public String writeToString(A a) {
		String out = null;
		if (a != null) {
			try {
				ByteArrayOutputStream baos = new ByteArrayOutputStream();
				ObjectOutputStream oos = new ObjectOutputStream(baos);
				oos.writeObject(a);
				out = DatatypeConverter.printBase64Binary(baos.toByteArray());
			} catch (IOException e) {
				e.printStackTrace();
				return null;
			}
		}
		return out;
	}

	/**
	 * Converts a string back to an object.
	 * 
	 * @param str
	 *            serialized object.
	 * @return unserialized object.
	 */
	@SuppressWarnings("unchecked")
	public A readStringToObject(String str) {
		Object out = null;
		if (str != null) {
			try {
				ByteArrayInputStream bios = new ByteArrayInputStream(DatatypeConverter.parseBase64Binary(str));
				ObjectInputStream ois = new ObjectInputStream(bios);
				out = ois.readObject();
			} catch (IOException e) {
				e.printStackTrace();
				return null;
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
				return null;
			}
		}
		return (A) out;
	}

	/**
	 * @return filename in which the object should be/is serialized.
	 */
	public String getFileName() {
		return fileName;
	}
}