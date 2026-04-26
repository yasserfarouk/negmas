package exampleagentstest;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;

import org.junit.Before;

import genius.core.exceptions.InstantiateException;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfoList;
import storageexample.GroupX;

public class StorageExampleTest extends AgentTest {

	@Override
	@Before
	public void before() throws InstantiateException, IOException {
		super.before();

		// persistent data neeeded for some tests.
		when(persistentData.getPersistentDataType())
				.thenReturn(PersistentDataType.STANDARD);

		StandardInfoList standardInfoList = mock(StandardInfoList.class);
		when(standardInfoList.isEmpty()).thenReturn(true);
		when(persistentData.get()).thenReturn(standardInfoList);
	}

	public StorageExampleTest() {
		super(GroupX.class);
	}

}
