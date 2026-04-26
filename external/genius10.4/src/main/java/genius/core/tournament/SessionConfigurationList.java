package genius.core.tournament;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

import genius.core.AgentID;
import genius.core.config.MultilateralTournamentConfigurationInterface;
import genius.core.list.AbstractImmutableList;
import genius.core.list.FlatList;
import genius.core.list.Function;
import genius.core.list.Function2;
import genius.core.list.ImArrayList;
import genius.core.list.ImmutableList;
import genius.core.list.JavaList;
import genius.core.list.JoinedList;
import genius.core.list.MapList;
import genius.core.list.MapThreadList;
import genius.core.list.Permutations;
import genius.core.list.PermutationsOrderedWithoutReturn;
import genius.core.list.PermutationsWithReturn;
import genius.core.list.PermutationsWithoutReturn;
import genius.core.list.ShuffledList;
import genius.core.list.Tuple;
import genius.core.list.Tuples;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.session.Participant;
import genius.core.session.SessionConfiguration;

/**
 * a list of {@link SessionConfiguration}s for running. immutable.
 *
 */
public class SessionConfigurationList extends AbstractImmutableList<SessionConfiguration> {

	private static final AgentID MEDIATOR_ID = new AgentID("mediator");

	private ImmutableList<SessionConfiguration> sessions = new ImArrayList<SessionConfiguration>();

	private MultilateralTournamentConfigurationInterface config;

	private Participant mediator;

	/**
	 * Constructs the list given a tournament configuration.
	 * 
	 * @param config
	 *            the configuration. Assumes the configuration is correct.
	 */
	public SessionConfigurationList(MultilateralTournamentConfigurationInterface config) {
		this.config = config;
		if (config.getProtocolItem().getHasMediator()) {
			this.mediator = new Participant(MEDIATOR_ID, config.getMediator(), config.getProfileItems().get(0));
		}
		for (int tournament = 0; tournament < config.getRepeats(); tournament++) {
			sessions = new JoinedList<SessionConfiguration>(sessions, getOneTournamentSessions());
		}
	}

	@Override
	public SessionConfiguration get(BigInteger index) {
		return sessions.get(index);
	}

	@Override
	public BigInteger size() {
		return sessions.size();
	}

	// *********** all code below this point works for 1 repeat ***********

	/**
	 * @return all sessions as specified in the config.
	 */
	private ImmutableList<SessionConfiguration> getOneTournamentSessions() {
		ImmutableList<SessionConfiguration> list = getOneNonrandomTournamentSessions();

		if (config.isRandomSessionOrder()) {
			list = new ShuffledList<>(list);
		}
		return list;

	}

	// *********** all code below this point does not randomize ***********

	/**
	 * @return All sessions as specified in the config,ignoring session
	 *         randomization.
	 */
	private ImmutableList<SessionConfiguration> getOneNonrandomTournamentSessions() {
		return isSpecialBilateral() ? getBilateralSessions() : getMultilateralSessions();

	}

	/**
	 * 
	 * @return true iff this config has separate parties and profiles for both
	 *         sides.
	 */
	private boolean isSpecialBilateral() {
		return config.getNumPartiesPerSession() == 2 && !config.getPartyBItems().isEmpty();
	}

	/**
	 * @return a set of bilateral sessions using different party-profile sets
	 *         for both sides.
	 */
	private ImmutableList<SessionConfiguration> getBilateralSessions() {
		ImArrayList<ParticipantRepItem> partiesA = new ImArrayList<>(config.getPartyItems());
		ImArrayList<ProfileRepItem> profilesA = new ImArrayList<>(config.getProfileItems());
		ImmutableList<Participant> sideA = new FlatList<>(getParticipants(partiesA, profilesA, 1, true));

		ImArrayList<ParticipantRepItem> partiesB = new ImArrayList<>(config.getPartyBItems());
		ImArrayList<ProfileRepItem> profilesB = new ImArrayList<>(config.getProfileBItems());
		ImmutableList<Participant> sideB = new FlatList<>(getParticipants(partiesB, profilesB, 1, true));

		return new MapList<>(new configFunc(), new Tuples<>(sideA, sideB));
	}

	/**
	 * @return a set of multilateral sessions using different party-profile sets
	 *         for both sides.
	 */
	private ImmutableList<SessionConfiguration> getMultilateralSessions() {
		ImmutableList<ParticipantRepItem> parties = new ImArrayList<>(config.getPartyItems());
		ImmutableList<ProfileRepItem> profiles = new ImArrayList<>(config.getProfileItems());

		ImmutableList<ImmutableList<Participant>> partieslist = getParticipants(parties, profiles,
				config.getNumPartiesPerSession(), config.isRepetitionAllowed());

		return new MapList<ImmutableList<Participant>, SessionConfiguration>(new myMultiConfig(), partieslist);

	}

	/**
	 * 
	 * @param parties
	 * @param profiles
	 * @param n
	 *            number of items to draw. Assumed >= 1.
	 * @param drawPartyWithPutback
	 *            if parties can be drawn multiple times
	 * @return all Permutations of parties with profiles. Profiles are drawn
	 *         with replace.
	 */
	private ImmutableList<ImmutableList<Participant>> getParticipants(ImmutableList<ParticipantRepItem> parties,
			ImmutableList<ProfileRepItem> profiles, int n, boolean drawPartyWithPutback) {

		Permutations<ParticipantRepItem> partiesPermutations;
		if (drawPartyWithPutback) {
			partiesPermutations = new PermutationsWithReturn<>(parties, n);
		} else {
			partiesPermutations = new PermutationsWithoutReturn<>(parties, n);
		}

		Permutations<ProfileRepItem> profilesPermutations = new PermutationsOrderedWithoutReturn<>(profiles, n);

		Tuples<ImmutableList<ParticipantRepItem>, ImmutableList<ProfileRepItem>> tuples = new Tuples<>(
				partiesPermutations, profilesPermutations);

		return new MapList<>(new partiesFromTuples(), tuples);
	}

	////////////////////// Basal helper functions. Could be one-liners in java8
	/**
	 * function mapping N participants into a SessionConfiguration.
	 */
	class myMultiConfig implements Function<ImmutableList<Participant>, SessionConfiguration> {
		@Override
		public SessionConfiguration apply(ImmutableList<Participant> parties) {
			return new SessionConfiguration(config.getProtocolItem(), mediator, new JavaList<>(parties),
					config.getDeadline(), config.getPersistentDataType());
		}
	}

	/**
	 * function that takes tuple < list of perties, list of profiles> and
	 * returns a list of parties.
	 */
	class partiesFromTuples implements
			Function<Tuple<ImmutableList<ParticipantRepItem>, ImmutableList<ProfileRepItem>>, ImmutableList<Participant>> {

		@Override
		public ImmutableList<Participant> apply(
				Tuple<ImmutableList<ParticipantRepItem>, ImmutableList<ProfileRepItem>> tuple) {
			return new MapThreadList<Participant, ParticipantRepItem, ProfileRepItem>(new fparty(), tuple.get1(),
					tuple.get2());
		}
	}

	/**
	 * Function mapping 2 participants in a sessionconfiguration.
	 */
	private class configFunc implements Function<Tuple<Participant, Participant>, SessionConfiguration> {
		@Override
		public SessionConfiguration apply(Tuple<Participant, Participant> tuple) {
			List<Participant> parties = new ArrayList<>();
			parties.add(tuple.get1());
			parties.add(tuple.get2());
			return new SessionConfiguration(config.getProtocolItem(), mediator, parties, config.getDeadline(),
					config.getPersistentDataType());
		}
	}

}

/**
 * simple function that creates participant
 */
class fparty implements Function2<ParticipantRepItem, ProfileRepItem, Participant> {
	private static int nr = 0;

	@Override
	public Participant apply(ParticipantRepItem party, ProfileRepItem profile) {
		return new Participant(new AgentID("Party" + (nr++)), party, profile);
	}
}
