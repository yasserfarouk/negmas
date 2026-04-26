package genius.core.config;

import java.io.File;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.xml.bind.JAXBException;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementRef;
import javax.xml.bind.annotation.XmlElementWrapper;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlSeeAlso;

import genius.core.Deadline;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.repository.boa.BoaPartyRepItem;

/**
 * Implementation of MultilateralTournamentConfigurationInterface. This stores
 * all information for a multilateral tournament. Can {@link #load(File)} and
 * {@link #save(File)} to XML. Immutable.
 * 
 * This object can also be deserialized. Therefore we set default values in
 * optional fields
 */
@XmlRootElement
@XmlAccessorType(XmlAccessType.FIELD)
@XmlSeeAlso({ BoaPartyRepItem.class, PartyRepItem.class })
public class MultilateralTournamentConfiguration
		implements MultilateralTournamentConfigurationInterface {

	@XmlElement(name = "deadline")
	private Deadline deadline;

	/**
	 * Holds the chosen protocol
	 */
	@XmlElement
	private MultiPartyProtocolRepItem protocolItem;

	/**
	 * The mediator in use. ignored if protocol does not use mediator. Must be
	 * non-null if protocol needs mediator.
	 */
	@XmlElement
	private PartyRepItem mediator;

	/**
	 * Holds the list of all chosen parties. Excludes mediator
	 */
	@XmlElementWrapper(name = "partyRepItems")
	@XmlElementRef
	private List<ParticipantRepItem> partyItems;

	/**
	 * Holds the list of chosen profiles
	 */
	@XmlElementWrapper(name = "partyProfileItems")
	@XmlElement(name = "item")
	private List<ProfileRepItem> profileItems;

	@XmlElementWrapper(name = "partyBItems")
	@XmlElementRef
	private List<? extends ParticipantRepItem> partyBItems = new ArrayList<>();

	@XmlElementWrapper(name = "partyBProfiles")
	@XmlElement(name = "item")
	private List<? extends ProfileRepItem> profileBItems = new ArrayList<>();

	/**
	 * Holds the number of "agents per session" as set in the GUI. This is the
	 * number of non-mediators in each session. This may be smaller than the
	 * number of available partyItems.
	 */
	@XmlElement
	private int numberOfPartiesPerSession;

	/**
	 * Holds whether repetition is allowed or not;
	 */
	@XmlElement
	private boolean repetitionAllowed;

	@XmlElement
	private boolean isRandomSessionOrder;

	@XmlElement
	private int repeats = 1;

	@XmlElement
	private PersistentDataType persistentDataType;

	@XmlElement
	private boolean enablePrint;

	/**
	 * Needed for the serializer.
	 */
	@SuppressWarnings("unused")
	private MultilateralTournamentConfiguration() {
	}

	public MultilateralTournamentConfiguration(
			MultiPartyProtocolRepItem protocol, Deadline deadline2,
			PartyRepItem mediator, List<ParticipantRepItem> parties,
			List<ProfileRepItem> profiles, List<ParticipantRepItem> partiesB,
			List<ProfileRepItem> profilesB, int nrepeats, int nparties,
			boolean repeatParties, boolean isRandomSessionOrder,
			PersistentDataType type, boolean enablePrint) {
		this.protocolItem = protocol;
		this.deadline = deadline2;
		this.mediator = mediator;
		this.partyItems = new ArrayList<>(parties);
		this.profileItems = new ArrayList<>(profiles);
		this.partyBItems = new ArrayList<>(partiesB);
		this.profileBItems = new ArrayList<>(profilesB);
		this.repeats = nrepeats;
		this.numberOfPartiesPerSession = nparties;
		this.repetitionAllowed = repeatParties;
		this.isRandomSessionOrder = isRandomSessionOrder;
		this.persistentDataType = type;
		this.enablePrint = enablePrint;

	}

	@Override
	public MultiPartyProtocolRepItem getProtocolItem() {
		return protocolItem;
	}

	@Override
	public PartyRepItem getMediator() {
		return mediator;
	}

	@Override
	public Deadline getDeadline() {
		return deadline;
	}

	@Override
	public List<ParticipantRepItem> getPartyItems() {
		return Collections.unmodifiableList(partyItems);
	}

	@Override
	public List<ProfileRepItem> getProfileItems() {
		return Collections.unmodifiableList(profileItems);
	}

	@Override
	public List<ParticipantRepItem> getPartyBItems() {
		return Collections.unmodifiableList(partyBItems);
	}

	@Override
	public List<ProfileRepItem> getProfileBItems() {
		return Collections.unmodifiableList(profileBItems);
	}

	@Override
	public int getRepeats() {
		return repeats;
	}

	@Override
	public int getNumPartiesPerSession() {
		return numberOfPartiesPerSession;
	}

	@Override
	public boolean isRepetitionAllowed() {
		return repetitionAllowed;
	}

	@Override
	public boolean isRandomSessionOrder() {
		return isRandomSessionOrder;
	}

	/**
	 * Load a new {@link MultilateralTournamentConfiguration} from file.
	 * 
	 * @param file
	 *            the file to load from
	 * @return the new {@link MultilateralTournamentConfiguration}.
	 * @throws JAXBException
	 */
	public static MultilateralTournamentConfiguration load(File file)
			throws JAXBException {
		return MultilateralTournamentsConfiguration.load(file).getTournaments()
				.get(0);
	}

	/**
	 * Save this to output
	 * 
	 * @param outstream
	 *            the outputstream to write to
	 */
	public void save(OutputStream outstream) {
		List<MultilateralTournamentConfiguration> tournaments = new ArrayList<MultilateralTournamentConfiguration>();
		tournaments.add(this);
		new MultilateralTournamentsConfiguration(tournaments).save(outstream);
	}

	@Override
	public PersistentDataType getPersistentDataType() {
		return persistentDataType;
	}

	@Override
	public boolean isPrintEnabled() {
		return enablePrint;
	}

}
