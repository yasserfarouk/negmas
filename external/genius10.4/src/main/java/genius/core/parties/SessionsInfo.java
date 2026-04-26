package genius.core.parties;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;

import genius.core.Global;
import genius.core.misc.FileTools;
import genius.core.persistent.PersistentDataType;
import genius.core.protocol.MultilateralProtocol;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;

/**
 * immutable info for all sessions.
 */
public class SessionsInfo {

	private Path storageDir;
	private final MultilateralProtocol protocol;
	private PersistentDataType persistentDataType;
	private boolean isPrintEnabled;

	public SessionsInfo(MultilateralProtocol protocol, PersistentDataType type, boolean isPrintEnabled)
			throws IOException {
		if (type == null)
			throw new NullPointerException("type");
		this.persistentDataType = type;
		this.protocol = protocol;
		this.storageDir = Files.createTempDirectory("GeniusData");
		this.isPrintEnabled = isPrintEnabled;
	}

	/**
	 * Try to return the stored data for given [party,profile] pair. If there is
	 * no file we return null. If there is a deserialization error, we delete
	 * the file and print an exception.
	 * 
	 * @param party
	 *            the {@link PartyRepItem}
	 * @param profile
	 *            the {@link ProfileRepItem}
	 * @return storage for given agent and profile.
	 */
	public Serializable getStorage(ParticipantRepItem party, ProfileRepItem profile)
			throws ClassNotFoundException, IOException {
		Path path = getPath(party, profile);
		if (!Files.exists(path)) {
			return null;
		}

		// file exists. Try to deserialize and return it
		try {
			return Global.deserializeObject(new FileInputStream(path.toFile()));
		} catch (ClassNotFoundException | IOException e) {
			try {
				Files.delete(path);
			} catch (IOException e1) {
				e1.printStackTrace();
			}
			throw e;
		}

	}

	/**
	 * @param party
	 *            the {@link PartyRepItem}
	 * @param profile
	 *            the {@link ProfileRepItem}
	 * @return the path to the temp file where the data is saved for given
	 *         agentID + profile
	 */
	public Path getPath(ParticipantRepItem party, ProfileRepItem profile) {
		return storageDir.resolve(party.getUniqueName() + "-" + profile.getFullName());
	}

	/**
	 * Closes the SessionsInfo: removes the tmp dir, deletes all saved files.
	 * This SessionsInfo can not be used after calling this.
	 */
	public void close() {
		if (storageDir != null) {
			FileTools.deleteDir(storageDir.toFile());
			storageDir = null;
		}
	}

	/**
	 * saves provided storageMap to the storageDir. The storageMap is just
	 * removed from disk if it's empty.
	 * 
	 * @param content
	 *            the data to save. If null, the old data is removed but no
	 *            'null' object is saved.
	 * @param party
	 *            the {@link PartyRepItem}
	 * @param profile
	 *            the {@link ProfileRepItem}
	 * @throws IOException
	 */
	public void saveStorage(Serializable content, ParticipantRepItem party, ProfileRepItem profile) throws IOException {

		Path path = getPath(party, profile);
		if (Files.exists(path)) {
			Files.delete(path);
		}
		if (content != null) {
			Global.serializeObject(new FileOutputStream(path.toFile()), content);
		}
	}

	public MultilateralProtocol getProtocol() {
		return protocol;
	}

	public PersistentDataType getPersistentDataType() {
		return persistentDataType;
	}

	/**
	 * True if print ot stdout is enabled.
	 * 
	 * @return
	 */
	public boolean isPrintEnabled() {
		return isPrintEnabled;
	}
}
