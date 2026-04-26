package genius.core.repository.boa;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;
import javax.xml.bind.annotation.XmlRootElement;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BOA;
import genius.core.boaframework.BoaType;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * A repository for BOA components. A BOA component repository is special
 * because it contains 4 lists instead of 1 list of rep items. immutable.
 *
 */
@XmlRootElement(name = "repository")
public class BoaRepository {

	@XmlElementWrapper(name = "biddingstrategies")
	@XmlElement(name = "biddingstrategy")
	BoaRepItemList<BoaRepItem<OfferingStrategy>> biddingStrategies = new BoaRepItemList<>();

	@XmlElementWrapper(name = "acceptanceconditions")
	@XmlElement(name = "acceptancecondition")
	BoaRepItemList<BoaRepItem<AcceptanceStrategy>> acceptanceConditions = new BoaRepItemList<>();

	@XmlElementWrapper(name = "opponentmodels")
	@XmlElement(name = "opponentmodel")
	BoaRepItemList<BoaRepItem<OpponentModel>> opponentModels = new BoaRepItemList<>();

	@XmlElementWrapper(name = "omstrategies")
	@XmlElement(name = "omstrategy")
	BoaRepItemList<BoaRepItem<OMStrategy>> opponentStrategies = new BoaRepItemList<>();

	/** The file where this repo is read/saved */
	private File file;

	private BoaRepository() {
	}

	public static BoaRepository loadRepository(File file) throws JAXBException {
		JAXBContext jaxbContext = JAXBContext.newInstance(BoaRepository.class);
		Unmarshaller unmarshaller = jaxbContext.createUnmarshaller();
		unmarshaller.setEventHandler(new javax.xml.bind.helpers.DefaultValidationEventHandler());
		BoaRepository rep = (BoaRepository) (unmarshaller.unmarshal(file));
		rep.file = file;
		return rep;
	}

	/**
	 * Save this repo to {@link #file}. @throws IOException @throws
	 */
	public void save() throws IOException {
		JAXBContext context;
		try {
			context = JAXBContext.newInstance(BoaRepository.class);
			Marshaller m = context.createMarshaller();
			m.marshal(this, new FileWriter(file));
		} catch (JAXBException e) {
			throw new IOException("failed to save " + file, e);
		}
	}

	public BoaRepItemList<BoaRepItem<OfferingStrategy>> getBiddingStrategies() {
		return biddingStrategies;
	}

	public BoaRepItemList<BoaRepItem<AcceptanceStrategy>> getAcceptanceConditions() {
		return acceptanceConditions;
	}

	public BoaRepItemList<BoaRepItem<OpponentModel>> getOpponentModels() {
		return opponentModels;
	}

	public BoaRepItemList<BoaRepItem<OMStrategy>> getOpponentModelStrategies() {
		return opponentStrategies;
	}

	@Override
	public String toString() {
		return "BoaRepository[" + biddingStrategies + "," + acceptanceConditions + "," + opponentModels + ","
				+ opponentStrategies + "]";
	}

	public void addComponent(BoaRepItem newComponent) {
		System.out.println("TODO add BOA component");
	}

	public void removeComponent(BoaRepItem removed) {
		System.out.println("TODO remove BOA component");

	}

	/**
	 * @param type
	 *            the needed {@link BoaType}
	 * @return BOA items of type. Must match T, and we can't check that.
	 * 
	 */
	@SuppressWarnings({ "rawtypes", "incomplete-switch", "unchecked" })
	public <T extends BOA> BoaRepItemList<BoaRepItem<T>> getBoaComponents(BoaType type) {
		// bit of nasty casting, since we can't prove here that T and type
		// match...
		switch (type) {
		case ACCEPTANCESTRATEGY:
			return (BoaRepItemList) getAcceptanceConditions();
		case BIDDINGSTRATEGY:
			return (BoaRepItemList) getBiddingStrategies();
		case OPPONENTMODEL:
			return (BoaRepItemList) getOpponentModels();
		case OMSTRATEGY:
			return (BoaRepItemList) getOpponentModelStrategies();
		}
		return null;
	}

}
