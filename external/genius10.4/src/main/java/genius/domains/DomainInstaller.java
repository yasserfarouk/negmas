package genius.domains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.CodeSource;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Installs domain xml files either from jar file or from file system (if you
 * run from eclipse). Nothing happens if /etc is already there.
 *
 */
public class DomainInstaller {

	/**
	 * Installs the domain files for use in Genius. Specifically, copies
	 * "resources/genius/repositories/domains/domainrepository.xml" in "."
	 * (current dir) and the entire directory "resources/genius/template" to
	 * "./etc/". Files that are already exist are not touched. This means that
	 * eg "./etc" already exists, nothing happens.
	 * 
	 * @throws IOException
	 *             if failes to copy the files
	 */
	public static void run() throws IOException {
		copyRecursively("genius/templates/", "etc/templates/");
		copyRecursively("genius/repositories/domains/domainrepository.xml", "domainrepository.xml");
	}

	/**
	 * 
	 * @param dir
	 *            the head part of all files needed, eg "genius/templates/".
	 * @param target
	 *            the head part to be used for the target files. We replace
	 *            "dir" with "target" in all matching files. Remember to add
	 *            trailing "/" if you have trailing "/" in dir as well
	 * @throws IOException
	 * @throws URISyntaxException
	 */
	protected static void copyRecursively(String dir, String target) throws IOException {
		// are we running from a jar?
		CodeSource src = DomainInstaller.class.getProtectionDomain().getCodeSource();
		if (src == null)
			throw new FileNotFoundException("code base not found");
		URL uri = src.getLocation();
		// file://... but the file may be a jar after all..
		if (uri.toString().endsWith("jar")) {
			copyJarSystem(dir, target, uri);
		} else {
			// assume it's a normal file
			copyFileSystem("/" + dir, target);
		}

	}

	/**
	 * Copy a resource inside a JAR recursively
	 * 
	 * @param dir
	 *            resource name, class path string like "/genius/repositories"
	 * @param target
	 *            target directory eg "test" (writes in current directory
	 *            "./test"). Do not end the target with "/". If you copy just a
	 *            file, give the exact target filename here.
	 * @throws IOException
	 *             if problem with reading or writing the file
	 * @throws URISyntaxException
	 *             if string can not be converted to valid file path
	 */
	private static void copyFileSystem(String dir, String target) throws IOException {

		final Path destpath = Paths.get(target);
		if (destpath.toFile().exists()) {
			return;
		}
		final URL url = DomainInstaller.class.getResource(dir);
		if (url == null) {
			throw new FileNotFoundException("Could not find resource " + dir);
		}
		try {
			final File file = new File(url.toURI());
			if (file.isDirectory()) {
				destpath.toFile().getParentFile().mkdirs();
				Files.copy(Paths.get(url.toURI()), destpath);
				for (File sub : file.listFiles()) {
					copyFileSystem(dir + "/" + sub.getName(), target + "/" + sub.getName());
				}
			} else {
				Files.copy(Paths.get(url.toURI()), Paths.get(target));
			}
		} catch (URISyntaxException e) {
			throw new IOException("Encountered wrong filename while copying resources", e);
		}

	}

	/**
	 * Copy all files recursively where the source is inside a jar.
	 * 
	 * @param dir
	 * @param target
	 * @param uri
	 * @throws IOException
	 */
	private static void copyJarSystem(String dir, String target, URL uri) throws IOException {
		final Path destpath = Paths.get(target);
		if (destpath.toFile().exists()) {
			return;
		}
		ZipInputStream zip = new ZipInputStream(uri.openStream());
		while (true) {
			ZipEntry e = zip.getNextEntry();
			if (e == null)
				break;
			String name = e.getName();
			if (name.startsWith(dir)) {
				copyResource("/" + name, name.replace(dir, target));
			}
		}
	}

	/**
	 * Copy 1 resource inside a jar. If given name actually ends with "/" it is
	 * assumed a directory and the directory is created if it does not yet
	 * exist.
	 * 
	 * @param name
	 *            the file/directory name.Name should end with "/" iff it is a
	 *            directory.
	 * @param target
	 *            the target name/directory.
	 * @throws IOException
	 */
	private static void copyResource(String name, String target) throws IOException {
		/*
		 * make the directory that should hold this file. Notice: sometimes the
		 * zip file contains the directories in "nice" order so that dir names
		 * are before files inside these dirs. And at other times, they are
		 * not...
		 */
		createDir(new File(target));
		if (name.endsWith("/")) {
			return;
		}

		InputStream stream = null;
		OutputStream resStreamOut = null;
		try {
			stream = DomainInstaller.class.getResourceAsStream(name);
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

	private static void createDir(File dir) {
		if (dir == null)
			return;
		if (!dir.isDirectory()) {
			dir = dir.getParentFile();
			if (dir == null)
				return;
		}
		dir.mkdirs();
	}

}