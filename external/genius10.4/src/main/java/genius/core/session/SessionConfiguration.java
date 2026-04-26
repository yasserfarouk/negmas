package genius.core.session;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import genius.core.Deadline;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.boa.BoaPartyRepItem;

/**
 * Holds all information to start and run a session. Contains only references,
 * no actual agent instantiations. Immutable implementation
 */
public class SessionConfiguration implements MultilateralSessionConfiguration {

	private MultiPartyProtocolRepItem protocol;
	private List<Participant> parties;
	private Deadline deadline;
	private Participant mediator;
	private PersistentDataType persistentDataType;

	public SessionConfiguration(MultiPartyProtocolRepItem protocol, Participant mediator, List<Participant> parties,
			Deadline deadline, PersistentDataType type) {
		this.protocol = protocol;
		this.parties = parties;
		this.deadline = deadline;
		this.mediator = mediator;
		this.persistentDataType = type;
	}

	@Override
	public MultiPartyProtocolRepItem getProtocol() {
		return protocol;
	}

	@Override
	public List<Participant> getParties() {
		return Collections.unmodifiableList(parties);
	}

	@Override
	public Deadline getDeadline() {
		return deadline;
	}

	@Override
	public String toString() {
		return "SessionConfiguration[" + protocol + "," + parties + "," + deadline + "]";
	}

	@Override
	public Participant getMediator() {
		return mediator;
	}

	@Override
	public PersistentDataType getPersistentDataType() {
		return persistentDataType;
	}

	/**
	 * @return simple list of names
	 */
	public List<String> getParticipantNames() {
		List<String> participantnames = new ArrayList<String>();
		for (Participant config : getParties()) {
			ParticipantRepItem repitem = config.getStrategy();
			String itemname;
			if (repitem instanceof BoaPartyRepItem) {
				itemname = ((BoaPartyRepItem) repitem).getUniqueName();
			} else if (repitem instanceof PartyRepItem) {
				itemname = ((PartyRepItem) repitem).getClassPath();
			} else {
				itemname = "unknown repitem " + repitem;
			}
			participantnames.add(itemname);
		}

		return participantnames;
	}

}
