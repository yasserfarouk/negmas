package genius.core.misc;

import java.io.File;

public class FileTools {

	/**
	 * Delete directory and all recursively all files contained in it.
	 * 
	 * @param file
	 *            directory
	 */
	public static void deleteDir(File file) {
		File[] contents = file.listFiles();
		if (contents != null) {
			for (File f : contents) {
				deleteDir(f);
			}
		}
		file.delete();
	}
}
