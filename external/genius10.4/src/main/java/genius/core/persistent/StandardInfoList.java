package genius.core.persistent;

import java.io.Serializable;
import java.util.List;

import genius.core.config.MultilateralTournamentConfigurationInterface;

/**
 * list of {@link StandardInfo} data, in the order of the sessions as they were
 * run in the current running
 * {@link MultilateralTournamentConfigurationInterface}, 0 being the first in
 * the tournament. The current running session is not included in the list.
 * immutable.
 */
public interface StandardInfoList extends List<StandardInfo>, Serializable {
}
