package genius.core.utility;

import java.io.File;
import java.io.Serializable;

import javax.swing.JOptionPane;

import genius.core.Agent;
import genius.core.SerializeHandling;

/**
 * This class handles saving and loading data for agents which are negotiating
 * in a specific preference profile. Every unique combination of an agent class
 * name and a preference profile name will be saved in a different file; for
 * identification.
 * 
 * This is here to support the old {@link Agent}. The files are just available
 * in the file system and therefore agents can read each other's data. Therefore
 * it is recommended that this is not used for new developments.
 */
public class DataObjects {

	/**
	 */
	/**
	 * Contains the absolute path of the source folder into which we would like
	 * to save the data of all the agents.
	 */
	private String absolutePath;

	/** A reference to the folder in the absolutePath */
	private File theFolder;
	/**
	 * The folder name where the data is stored.
	 */
	private final String dataFolderName = "DataObjects";

	private static DataObjects instance = new DataObjects();

	private DataObjects() {
		absolutePath = new File(".").getAbsolutePath();
		absolutePath = absolutePath.replace('\\', '/');
		absolutePath = absolutePath + "/" + dataFolderName;
		theFolder = new File(absolutePath);
		if (!theFolder.exists()) {
			createFolder();
		} else {
			restartFolder();
		}
	}

	public static DataObjects getInstance() {
		return instance;
	}

	/**
	 * Restarts the folder "theFolder", meaning it deletes all files in it and
	 * then creates a new empty folder with the same name.
	 * 
	 * @return true if succeeded
	 */
	public boolean restartFolder() {
		boolean ans = this.deleteFolderRecursively(theFolder);
		ans = ans && this.createFolder();
		return ans;
	}

	/**
	 * Creates a new empty folder from "theFolder" file.
	 * 
	 * @return true if succeeded
	 */
	private boolean createFolder() {
		try {
			theFolder.mkdir();
			return true;
		} catch (SecurityException e) {
			String msg = "Could not createFrom the folder in " + absolutePath;
			JOptionPane.showMessageDialog(null, msg,
					"Error while creating a folder ", 0);
		}
		return false;
	}

	/**
	 * Deletes all files and directories inside folder, and itself.
	 * 
	 * @param folder
	 *            the folder to be erased recursively
	 * @return true if succeeded
	 */
	private boolean deleteFolderRecursively(File folder) {
		File[] files = folder.listFiles();
		if (files != null) { // some JVMs return null for empty dirs
			for (File f : files) {
				if (f.isDirectory()) {
					deleteFolderRecursively(f);
				} else {
					f.delete();
				}
			}
		}
		return folder.delete();
	}

	/**
	 * Saves dataToSave of the agent with class agentClassName for a preference
	 * profile prefProfName. It creates a unique name (concatenating the agent
	 * class name and the preference profile file name) to be the name of the
	 * file which will contain dataToSave.
	 * 
	 * @param dataToSave
	 *            a {@link Serializable} object to save
	 * @param agentClassName
	 *            is the class name of the agent who wants to save the data
	 * @param prefProfName
	 *            is the preference profile for which the agent wants to save
	 *            data
	 * @return true if dataToSave was saved successfully false otherwise.
	 */
	public boolean saveData(Serializable dataToSave, String agentClassName,
			String prefProfName) {
		if (!theFolder.exists()) {
			createFolder();
		}
		String key = agentClassName + "_" + prefProfName;
		key = key.replace('/', '_');
		String path = absolutePath + "/" + key;
		return SerializeHandling.writeToDisc(dataToSave, path);
	}

	/**
	 * Loads the data of the agent, by the agentClassName and the prefProfName.
	 * It identifies the data by a unique name (concatenating the agent class
	 * name and the preference profile file name) which is the name of the file
	 * containing the data.
	 * 
	 * @param agentClassName
	 *            is the class name of the agent who wants to load the data
	 * @param prefProfName
	 *            is the preference profile for which the agent wants to load
	 *            the data
	 * @return the data that was saved by the agent when he was in the role of
	 *         the preference profile filename.
	 */
	public Serializable loadData(String agentClassName, String prefProfName) {
		String key = null;
		key = agentClassName + "_" + prefProfName;
		key = key.replace('/', '_');
		String path = absolutePath + "/" + key;
		return SerializeHandling.readFromDisc(path);
	}

}
