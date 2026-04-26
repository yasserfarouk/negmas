package genius.core.parties;

import java.util.List;

import java.util.ArrayList;
import java.util.Map;

import java.io.IOException;
import java.io.Serializable;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.exceptions.NegotiatorException;
import genius.core.list.ReadonlyList;
import genius.core.list.Tuple;
import genius.core.persistent.DefaultPersistentDataContainer;
import genius.core.persistent.DefaultStandardInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.session.RepositoryException;
import genius.core.session.Session;
import genius.core.timeline.Timeline;
import genius.core.uncertainty.UNCERTAINTYTYPE;
import genius.core.uncertainty.UncertainPreferenceContainer;
import genius.core.uncertainty.User;
import genius.core.uncertainty.UserModel;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.UncertainAdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;

/**
 * Only for use in the core. Keeps a NegotiationParty along with core-private
 * information.
 * 
 * @author W.Pasman 21jul15
 */
public class NegotiationPartyInternal implements PartyWithUtility {

	private NegotiationParty party;
	private UtilitySpace utilitySpace;
	private Session session;
	/**
	 * ID that should be unique for this party.
	 */
	private AgentID ID;
	private SessionsInfo sessionsInfo;
	private ProfileRepItem profileRepItem;
	private DefaultPersistentDataContainer storageMap;
	private ParticipantRepItem partyRepItem;
	private UncertainPreferenceContainer uncertainModel;
	private User user;

	/**
	 * Creates a new {@link NegotiationParty} from repository items and
	 * initializes it.
	 * 
	 * @param partyRepItem
	 *            the party reference
	 * @param profileRepItem
	 *            the profile to use for this party
	 * @param session
	 *            the session in which this runs
	 * @param agentID
	 *            the unique agentId to use, or null. If null, a unique ID will
	 *            be generated. For all default implementations, this has either
	 *            the format "ClassName" if only one such an agent exists (in
	 *            case of mediator for example [mediator always has the name
	 *            "mediator"]), "Party N" or it has the format "ClassName@N"
	 *            with N a unique integer if multiple agents of the same type
	 *            can exists.
	 * @param info
	 *            a SessionsInfo object that contains info shared with other
	 *            sessions.
	 * @throws RepositoryException
	 * @throws NegotiatorException
	 */
	public NegotiationPartyInternal(ParticipantRepItem partyRepItem,
			ProfileRepItem profileRepItem, Session session, SessionsInfo info,
			AgentID agentID) throws RepositoryException, NegotiatorException {
		if (agentID == null) {
			throw new NullPointerException("agentID");
		}
		this.session = session;
		this.sessionsInfo = info;
		this.ID = agentID;
		this.uncertainModel = uncertainModel;
		init(partyRepItem, profileRepItem, session, agentID);
	}

	/**
	 * @return the agent implementation
	 */
	public NegotiationParty getParty() {
		return party;
	}

	public double getUtility(Bid bid) {
		try {
			// throws exception if bid incomplete or not in utility space
			return bid == null ? 0 : utilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		}
	}

	public double getUtilityWithDiscount(Bid bid) {
		if (bid == null) {
			// utility is null if no bid
			return 0;
		} else if (session.getTimeline() == null) {
			// return undiscounted utility if no timeline given
			return getUtility(bid);
		} else {
			// otherwise, return discounted utility
			return utilitySpace.discount(utilitySpace.getUtility(bid),
					session.getTimeline().getTime());
		}

	}

	/**
	 * Gets the agent's utility space.
	 *
	 * @return the agent's utility space
	 */
	@Override
	public UtilitySpace getUtilitySpace() {
		return utilitySpace;
	}

	/**
	 * Gets the timeline for this agent.
	 *
	 * @return The timeline object or null if no timeline object (no time
	 *         constraints) set
	 */
	public Timeline getTimeLine() {
		return session.getTimeline();
	}

	/**
	 * Get the session that this party is using.
	 * 
	 * @return {@link Session}.
	 */
	public Session getSession() {
		return session;
	}

	@Override
	public String toString() {
		return ID.toString();
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((party == null) ? 0 : party.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		NegotiationPartyInternal other = (NegotiationPartyInternal) obj;
		if (party == null) {
			if (other.party != null)
				return false;
		} else if (!party.equals(other.party))
			return false;
		return true;
	}

	/************************* support functions ******************/

	/**
	 * Creates a new {@link NegotiationParty} from repository items and
	 * initializes it. This call is not sandboxed.
	 *
	 * @param partyRepItem
	 *            Party Repository item to createFrom party from
	 * @param profileRepItem
	 *            Profile Repository item to createFrom party from
	 * @return new Party
	 * @throws RepositoryException
	 * @throws java.lang.NoSuchMethodException
	 *             If requested Party does not have a constructor accepting only
	 *             preference profiles
	 * @throws java.lang.ClassNotFoundException
	 *             If requested Party class can not be found.
	 */
	private NegotiationParty init(ParticipantRepItem partyRepItem,
			ProfileRepItem profileRepItem, Session session, AgentID agentID)
			throws RepositoryException, NegotiatorException {
		this.profileRepItem = profileRepItem;
		this.partyRepItem = partyRepItem;
		this.utilitySpace = profileRepItem.create();
		if (!(utilitySpace instanceof AbstractUtilitySpace)) {
			throw new IllegalArgumentException("utilityspace in "
					+ profileRepItem
					+ " is not extending AbstractUtilitySpace and currently not supported for negotiation parties");
		}
		String err = utilitySpace.isComplete();
		if (err != null) {
			throw new IllegalArgumentException("utilityspace in "
					+ profileRepItem + " is not ready to run:" + err);
		}
		long randomSeed = System.currentTimeMillis();
		try {
			party = partyRepItem.load();
		} catch (Exception e) {
			throw new RepositoryException("failed to load " + partyRepItem, e);
		}

		getPersistentData(partyRepItem, profileRepItem);
		
		// Either a copy of the utility space is passed, or the estimate of the user model
		if (utilitySpace instanceof UncertainAdditiveUtilitySpace)
		{
			this.uncertainModel = new UncertainPreferenceContainer((UncertainAdditiveUtilitySpace) utilitySpace,
					UNCERTAINTYTYPE.PAIRWISECOMP);
			
			UserModel pairwiseCompUserModel = this.uncertainModel.getPairwiseCompUserModel();
			
			this.user = new User((UncertainAdditiveUtilitySpace) utilitySpace);
			// put into NegoInfo
			
			// estimate the utility space for any NegotiationParty for backwards compatibility
			// For an AbstractNegotiationParty, this is overriden in the init
			AbstractUtilitySpace passedUtilitySpace = AbstractNegotiationParty.defaultUtilitySpaceEstimator(utilitySpace.getDomain(), pairwiseCompUserModel);
			passedUtilitySpace.setReservationValue(utilitySpace.getReservationValue());
			passedUtilitySpace.setDiscount(((AbstractUtilitySpace) utilitySpace).getDiscountFactor());
			
			party.init(new NegotiationInfo(passedUtilitySpace,
					pairwiseCompUserModel, user,
					session.getDeadlines(), session.getTimeline(), randomSeed,
					agentID, storageMap));
		} 
		else 
		{
			// Otherwise, pass a copy of the utility space to the agent to prevent modification, with a null user model
			party.init(new NegotiationInfo(
					(AbstractUtilitySpace) utilitySpace.copy(), null, null,
					session.getDeadlines(), session.getTimeline(), randomSeed,
					agentID, storageMap));

		}
		return party;
	}

	/**
	 * Try to get the persistent storage data.
	 * 
	 * @param partyRepItem
	 * @param profileRepItem
	 */
	private void getPersistentData(ParticipantRepItem partyRepItem,
			ProfileRepItem profileRepItem) {
		PersistentDataType type = sessionsInfo.getPersistentDataType();
		Serializable data = null;
		switch (type) {
		case SERIALIZABLE:
		case STANDARD:
			try {
				data = sessionsInfo.getStorage(partyRepItem, profileRepItem);
			} catch (Exception e) {
				e.printStackTrace();
			}
			break;
		default:
			data = null;
			break;
		}
		if (data == null && type == PersistentDataType.STANDARD) {
			data = new DefaultStandardInfoList();
		}

		storageMap = new DefaultPersistentDataContainer(data, type);
	}

	/**
	 * Saves the persistent storage , see {@link SessionsInfo}.
	 * 
	 * @param actions
	 *            the actions that have been done in the last session. This is
	 *            added to the data first, if possible.
	 * @param agreement
	 *            agreement information: Bid and utility of the bid for this
	 *            party. Null if no agreement was reached.
	 * @profiles a list of [AgentID, profile-of-agent] names.
	 * 
	 * @throws IOException
	 */
	public void saveStorage(List<Action> actions, Map<String, String> profiles,
			Tuple<Bid, Double> agreement) throws IOException {
		if (storageMap.getPersistentDataType() == PersistentDataType.DISABLED)
			return;
		if (storageMap.getPersistentDataType() == PersistentDataType.STANDARD) {
			// update the data. bit hacky
			String startingagent = actions.isEmpty() ? "-"
					: actions.get(0).getAgent().toString();
			DefaultStandardInfoList data = (DefaultStandardInfoList) storageMap
					.get();
			List<Tuple<String, Double>> utilities = new ArrayList<>();
			for (Action action : actions) {
				if (action instanceof ActionWithBid) {
					utilities.add(new Tuple<String, Double>(
							action.getAgent().toString(),
							getUtility(((ActionWithBid) action).getBid())));
				}
			}

			data.addInternal(new DefaultStandardInfo(profiles, startingagent,
					utilities, session.getDeadlines(), agreement));
		}

		sessionsInfo.saveStorage(storageMap.get(), partyRepItem,
				profileRepItem);
	}

	/**
	 * 
	 * @return true iff htis party is a {@link Mediator}.
	 */
	public boolean isMediator() {
		return party instanceof Mediator;
	}

	public boolean isUncertain() {
		if (uncertainModel != null)
			return true;
		else
			return false;
	}

	@Override
	public AgentID getID() {
		return ID;
	}

	/**
	 * 
	 * @return {@link UncertainPreferenceContainer} or null if the profile was
	 *         not an {@link UncertainProfileRepItem}.
	 */
	public UncertainPreferenceContainer getUncertainModel() {
		return uncertainModel;
	}

	/**
	 * 
	 * @return {@link User} or null if the profile was
	 *         not an {@link User}.
	 */
	public User getUser() {
		return user;
	}
	
	/**
	 * Inner class, to prevent others casting to this. static to prevent
	 * serializer to serialize the superclass.
	 */
	@SuppressWarnings("serial")
	static class DefaultStandardInfoList extends ReadonlyList<StandardInfo>
			implements StandardInfoList, Serializable {

		public DefaultStandardInfoList(List<StandardInfo> infos) {
			super(infos);
		}

		public DefaultStandardInfoList() {
			super(new ArrayList<StandardInfo>());
		}

		/**
		 * backdoor for the usual {@link #add(StandardInfo)} this can be called
		 * only from the encapculating class.
		 * 
		 * @param e
		 *            the item to add to the list.
		 */
		public void addInternal(StandardInfo e) {
			list.add(e);
		}
	}

}
