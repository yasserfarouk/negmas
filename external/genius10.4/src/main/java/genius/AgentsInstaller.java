package genius;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * installs the xml files that Genius core needs in order to find back the
 * agents in this project.
 */
public class AgentsInstaller {
	/**
	 * Installs the agent xml files for use in Genius. Specifically, copies
	 * "resources/genius/repositories/domains/boapartyrepo.xml" and
	 * "resources/genius/repositories/parties/partyrepository.xml" to ".". Files
	 * that are already exist are not touched. This means that eg "./etc"
	 * already exists, nothing happens.
	 * 
	 * @throws IOException
	 *             if fails to copy the files
	 */
	public static void run() throws IOException {
		copy("genius/repositories/boa/boapartyrepo.xml");
		copy("genius/repositories/boa/boarepository.xml");
		copy("genius/repositories/parties/partyrepository.xml");

	}

	/**
	 * Copy file in given repository to the current directory.
	 * 
	 * @param filename
	 *            the filename in our repository
	 * @throws IOException
	 *             if copy fails
	 */
	private static void copy(String filename) throws IOException {
		// copy only takes the name but uses current directory
		File copy = new File(new File(filename).getName());
		if (copy.exists())
			return;
		copyResource(filename, copy.getName());
	}

	/**
	 * Copy of DomainInstaller copyResource... Copy 1 resource inside a jar. If
	 * given name actually ends with "/" it is assumed a directory and the
	 * directory is created if it does not yet exist.
	 * 
	 * @param name
	 *            the file/directory name.Name should end with "/" iff it is a
	 *            directory. Resourcess are assumed to be absolute refs, a
	 *            leading "/" is added before trying to resolve.
	 * @param target
	 *            the target name/directory. Usually does not start with "/"
	 *            because that would imply the root of the file system while
	 *            installs are usually done in the current directory.
	 * @throws IOException
	 *             if copy fails
	 */
	private static void copyResource(String name, String target) throws IOException {
		if (name.endsWith("/")) {
			new File(target).mkdirs();
			return;
		}
		InputStream stream = null;
		OutputStream resStreamOut = null;
		try {
			stream = AgentsInstaller.class.getResourceAsStream("/" + name);
			if (stream == null) {
				throw new FileNotFoundException("file not found " + name);
			}
			int readBytes;
			byte[] buffer = new byte[4096];
			resStreamOut = new FileOutputStream(target);
			while ((readBytes = stream.read(buffer)) > 0) {
				resStreamOut.write(buffer, 0, readBytes);
			}
		} finally {
			if (stream != null) {
				stream.close();
			}
			if (resStreamOut != null) {
				resStreamOut.close();
			}
		}
	}

}
