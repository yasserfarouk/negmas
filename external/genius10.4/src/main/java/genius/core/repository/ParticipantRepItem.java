package genius.core.repository;

import java.util.ArrayList;
import java.util.List;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;
import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;

import genius.core.exceptions.InstantiateException;
import genius.core.parties.NegotiationParty;
import genius.core.repository.boa.BoaPartyRepItem;

/**
 * Abstract superclass for RepItems that can generate Participants.
 *
 */
@SuppressWarnings("serial")
@XmlSeeAlso({ BoaPartyRepItem.class, PartyRepItem.class })
@XmlType
public abstract class ParticipantRepItem implements RepItem {

	/**
	 * true iff the cached elemenents have been initialized. Only
	 * the @XmlElement objects are set hard, the others are set when needed.
	 */
	private boolean initialized = false;

	/**
	 * Unused but still in some XML files. Maybe needed in future?
	 */
	@XmlElementWrapper(name = "properties")
	@XmlElement(name = "property")
	private List<String> properties = new ArrayList<String>();

	/**
	 * The protocol that this item supports. file path including the class name.
	 * cached.
	 */
	private String protocolClassPath = null;

	// deserialization support
	protected ParticipantRepItem() {
	}

	/**
	 * @return the actual {@link NegotiationParty} that this refers to.
	 * @throws InstantiateException
	 *             if party can't be loaded
	 * 
	 */
	public abstract NegotiationParty load() throws InstantiateException;

	public Boolean isMediator() {
		return false;
	}

	@Override
	public int hashCode() {
		return 0;
	}

	/**
	 * 
	 * @return a unique name that identifies this participant. Used e.g. to
	 *         create a storage space. Can be derived from the class name or
	 *         other settings in the participant. Notice that {@link #getName()}
	 *         may not be unique.
	 */
	public abstract String getUniqueName();

	public String getProtocolClassPath() {
		try {
			initSilent();
		} catch (InstantiateException e) {
			return e.toString();
		}
		return protocolClassPath;
	}

	/**
	 * Init our fields to cache the party information. party must have been set
	 * before getting here.
	 * 
	 * @param party
	 * @throws InstantiateException
	 */
	protected NegotiationParty init() throws InstantiateException {
		NegotiationParty party1 = load();
		protocolClassPath = party1.getProtocol().getCanonicalName();
		initialized = true;
		return party1;
	}

	/**
	 * call init but suppress any exceptions and print just a stacktrace.
	 * 
	 * @throws InstantiateException
	 */
	final protected void initSilent() throws InstantiateException {
		if (initialized)
			return;
		init();
	}

	/**
	 * @return a descriptor that is unique for this rep item class. Typically a
	 *         full.class.path but may be a composite.
	 */
	abstract public String getClassDescriptor();
}
