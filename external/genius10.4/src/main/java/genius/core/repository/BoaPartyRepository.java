package genius.core.repository;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;

import genius.core.listener.DefaultListenable;
import genius.core.repository.boa.BoaPartyRepItem;

/**
 * A list of {@link BoaPartyRepItem}s. This object is listenable for changes and
 * saves changes to given file.
 * 
 */
public class BoaPartyRepository extends DefaultListenable<BoaPartyRepItem> {

	private static final File REPO_FILE = new File("boapartyrepo.xml");
	private BoaPartiesList list;

	public BoaPartyRepository() throws FileNotFoundException, JAXBException {
		list = load();
	}

	/**
	 * Constructs a repo from file
	 * 
	 * @param repofile
	 *            the file to read the repo from
	 * @return a {@link BoaPartyRepository}.
	 * @throws JAXBException
	 * @throws FileNotFoundException
	 */
	private BoaPartiesList load() throws JAXBException, FileNotFoundException {
		JAXBContext jaxbContext = JAXBContext.newInstance(BoaPartiesList.class);
		Unmarshaller unmarshaller = jaxbContext.createUnmarshaller();
		return (BoaPartiesList) unmarshaller.unmarshal(new FileReader(REPO_FILE));
	}

	/**
	 * save the repo to disk.
	 * 
	 * @throws JAXBException
	 * @throws IOException
	 */
	protected void save() throws JAXBException, IOException {
		JAXBContext context = JAXBContext.newInstance(BoaPartiesList.class);
		Marshaller m = context.createMarshaller();
		m.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
		m.marshal(list, new FileWriter(REPO_FILE));
	}

	private void saveAndNotify(BoaPartyRepItem party) {
		try {
			save();
		} catch (JAXBException | IOException e) {
			// do not throw here. Nobody can recover from it.
			e.printStackTrace();
		}
		notifyChange(party);
	}

	/**
	 * Add party to the list
	 * 
	 * @param party
	 *            the {@link BoaPartyRepItem} to add.
	 */
	public void add(BoaPartyRepItem party) {
		list.add(party);
		saveAndNotify(party);
	}

	/**
	 * Remove boa party from the list
	 * 
	 * @param party
	 *            the party to remove
	 */
	public void remove(BoaPartyRepItem party) {
		list.remove(party);
		saveAndNotify(party);
	}

	public BoaPartiesList getList() {
		return list;
	}

	public void addAll(List<BoaPartyRepItem> values) {
		if (!values.isEmpty()) {
			list.addAll(values);
			saveAndNotify(values.get(0));
		}

	}

}
