"""
Qualtrics custom API wrapper for automatically generating the surveys for the STEL benchmark project
    https://www-annualreviews-org.proxy.library.uu.nl/doi/abs/10.1146/annurev-statistics-042720-125902
    https://github.com/60decibels/pytrics/blob/91ab8107433be0391f0fad06fa702eaa93f6af21/pytrics/qualtrics_api/client.py#L251
"""
import json
import logging
import qualtrics_constants
import pandas as pd
from STEL.utility import set_for_global

from STEL.utility.qualtrics_constants import EMBED_SCREENED_OUT, QID_PROLIFIC_PID, RESPONSE_TYPE_COL, VALID_RESPONSE
from STEL.utility.set_for_global import ALTERNATIVE11_COL, ID_COL, IN_SUBSAMPLE_COL
# personal constants for qualtrics usage -- need to be set to work
from z_personal_const import QUALTRICS_API_AUTH_TOKEN, QUALTRICS_API_BASE_URL, TEST_SURVEY_ID, EOSRedirectURL

BRANCH_CONJUCTION = "Conjuction"

QUADRUPLE = 'Quadruple'
TRIPLE = 'Triple'

QUESTIONS_PER_PAGE = 1

MULTIPLE_CHOICE_SUBSELECTOR = 'TX'

SCREEN_QS_FILENAME = '../test/fixtures/survey_base/screen_questions.tsv'
QUAD_QS_FILENAME = '../test/fixtures/survey_base/quad_questions_simplicity-formality-.tsv'

TEXT_ONLY_SELECTOR = 'TB'
TEXT_ONLY_TYPE = 'DB'

MULTIPLE_CHOICE_SELECTOR = "SAVR"
MULTIPLE_CHOICE_TYPE = "MC"

DRAG_RANK_SELECTOR = "DragAndDrop"
DRAG_RANK_TYPE = "PGR"
DRAG_RANK_SUBSELECTOR = "NoColumns"  # "Columns"

TEXT_ENTRY_TYPE = "TE"
TEXT_ENTRY_SELECTOR = "SL"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import requests
# import pytrics
import re


class QualtricsAPI:
    def __init__(self):
        self.auth_token = QUALTRICS_API_AUTH_TOKEN
        self.base_api_url = QUALTRICS_API_BASE_URL
        self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS = 'survey-definitions'
        self.QUALTRICS_API_PATH_BLOCKS = 'blocks'
        self.qualtrics_api_path_questions = 'questions'
        self.QUALTRICS_API_PATH_SURVEYS = 'surveys'
        self.QUALTRICS_API_PATH_SURVEY_OPTIONS = 'options'
        self.QUALTRICS_API_PATH_EXPORT_RESPONSES = 'export-responses'
        self.QUALTRICS_API_PATH_EMBEDDEDDATAFIELDS = 'embeddeddatafields'
        self.QUALTRICS_API_PATH_FLOW = 'flow'
        self._survey_id_regex = re.compile('^SV_[a-zA-Z0-9]{11,15}$')
        self._question_id_regex = re.compile('^QID[a-zA-Z0-9]+$')

        self._qualtrics_api_required_question_payload_keys = [
            'QuestionText', 'DataExportTag', 'QuestionType', 'Selector', 'Configuration',
            'QuestionDescription', 'Validation', 'Language']
        self._qualtrics_api_required_question_param_keys = [
            'text', 'tag_number', 'type', 'answer_selector',
            'label', 'is_mandatory', 'translations',
            'block_number']

        self._block_id_regex = re.compile('^BL_[a-zA-Z0-9]{11,15}$')
        self._qualtrics_api_supported_question_types = [
            'MC', 'Matrix', 'Captcha', 'CS', 'DB', 'DD', 'Draw', 'DynamicMatrix', 'FileUpload', 'GAP',
            'HeatMap', 'HL', 'HotSpot', 'Meta', 'PGR', 'RO', 'SBS', 'Slider', 'SS', 'TE', 'Timing',
        ]
        self._qualtrics_api_supported_question_types = [
            'MC', 'Matrix', 'Captcha', 'CS', 'DB', 'DD', 'Draw', 'DynamicMatrix', 'FileUpload', 'GAP',
            'HeatMap', 'HL', 'HotSpot', 'Meta', 'PGR', 'RO', 'SBS', 'Slider', 'SS', 'TE', 'Timing',
        ]
        self._qualtrics_api_supported_answer_selectors = [
            'DL', 'GRB', 'MACOL', 'MAHR', 'MAVR', 'ML', 'MSB', 'NPS',
            'SACOL', 'SAHR', 'SAVR', 'SB', 'TB', 'TXOT', 'PTB', 'SL',
            'DragAndDrop'
        ]

        self.QUALTRICS_API_SUPPORTED_DISPLAY_LOGIC_CHOICE_LOCATORS = [None, 'SelectableChoice']
        self.QUALTRICS_API_SUPPORTED_DISPLAY_LOGIC_CONJUNCTIONS = [None, 'Or', 'And']
        self.QUALTRICS_API_SUPPORTED_DISPLAY_LOGIC_OPERATORS = ['EqualTo', 'Selected', 'Is', 'NotSelected']
        self.QUALTRICS_API_SUPPORTED_LANGUAGE_CODES = [
            'AR', 'ASM', 'AZ-AZ', 'BEL', 'BG', 'BN', 'BS', 'CA', 'CEB', 'CH', 'CS', 'CY', 'DA', 'DE',
            'EL', 'EN-GB', 'EN-US', 'EN', 'EO', 'ES-ES', 'ES', 'ET', 'FA', 'FI', 'FR-CA', 'FR', 'GU',
            'HE', 'HE-ZA', 'HI', 'HIL', 'HR', 'HU', 'HYE', 'ID', 'ISL', 'IT', 'JA', 'KAN', 'KAT', 'KAZ',
            'KM', 'KO', 'LT', 'LV', 'MAL', 'MAR', 'MK', 'MN', 'MS', 'MY', 'NL', 'NO', 'ORI', 'PA-IN',
            'PL', 'PT-BR', 'PT', 'RO', 'RU', 'SIN', 'SK', 'SL', 'SQI', 'SR-ME', 'SR', 'SV', 'SW', 'TA',
            'TEL', 'TGL', 'TH', 'TR', 'UK', 'UR', 'VI', 'ZH-S', 'ZH-T',
        ]

        self.QUALTRICS_API_SUPPORTED_BLOCK_TYPES = ['Standard', 'Default', 'Trash']
        self.QUALTRICS_API_BLOCK_VISIBILITY_EXPANDED = 'Expanded'
        self.QUALTRICS_API_BLOCK_VISIBILITY_COLLAPSED = 'Collapsed'

    def _build_headers(self, method):
        """
        Constructs a dictionary which will be used as the request headers for all API interactions
        """
        if method not in ['GET', 'DELETE', 'POST', 'PUT']:
            raise Exception('Client only supports GET, DELETE, POST and PUT methods.')

        headers = {
            'X-API-TOKEN': self.auth_token,
        }

        if method in ['POST', 'PUT']:
            headers['Content-Type'] = 'application/json'

        return headers

    def _validate_survey_id(self, survey_id):
        survey_id_match = self._survey_id_regex.match(survey_id)
        if not survey_id_match:
            raise AssertionError('The format of survey_id is incorrect.')

    def _validate_block_id(self, block_id):
        block_id_match = self._block_id_regex.match(block_id)
        if not block_id_match:
            raise AssertionError('The format of block_id is incorrect.')

    def _validate_question_payload(self, question_payload=None):
        if not question_payload:
            raise AssertionError('The question payload is faulty.')

        missing_keys = []
        for required_key in self._qualtrics_api_required_question_payload_keys:
            if required_key not in question_payload.keys():
                missing_keys.append(required_key)

        if missing_keys:
            raise AssertionError('The question payload is invalid, keys missing: {}'.format(missing_keys))

        if 'Choices' in question_payload.keys():
            if 'ChoiceOrder' not in question_payload.keys():
                raise AssertionError('The question payload has Choices but no ChoiceOrder')

        if 'ChoiceOrder' in question_payload.keys():
            if 'Choices' not in question_payload.keys():
                raise AssertionError('The question payload has ChoiceOrder but no Choices')

        question_type = question_payload['QuestionType']
        selector = question_payload['Selector']

        if question_type == 'MC' and selector in ['SAVR', 'SAHR']:
            sub_selector = None

            if 'SubSelector' not in question_payload.keys():
                raise AssertionError('SubSelector missing from payload when expected')

            sub_selector = question_payload['SubSelector']

            if sub_selector != 'TX':
                raise AssertionError('The sub_selector: {} is invalid for question_type: {} and selector: {}'.format(
                    sub_selector,
                    question_type,
                    selector
                ))

    def _validate_question_id(self, question_id):
        question_id_match = self._question_id_regex.match(question_id)
        if not question_id_match:
            raise AssertionError('The format of question_id is incorrect.')

    def _find_question_in_survey_by_label(self, survey_id, question_label):
        survey_json = self._get_survey(survey_id)
        survey_dict = survey_json['result']

        for key, value in survey_dict['questions'].items():
            if value['questionLabel'] == question_label:
                return key, value

        return None, None

    def _build_question_display_logic(self, controlling_question_ids, choices, operators, conjunctions,
                                      locators):  # pylint: disable=too-many-arguments
        try:
            assert controlling_question_ids
            assert choices
            assert len(controlling_question_ids) == len(choices) == len(operators) == len(conjunctions) == len(locators)
            for operator in operators:
                assert operator.strip() in self.QUALTRICS_API_SUPPORTED_DISPLAY_LOGIC_OPERATORS
            for conjunction in conjunctions:
                assert conjunction in self.QUALTRICS_API_SUPPORTED_DISPLAY_LOGIC_CONJUNCTIONS
            for locator in locators:
                assert locator in self.QUALTRICS_API_SUPPORTED_DISPLAY_LOGIC_CHOICE_LOCATORS
        except (AssertionError, AttributeError):
            raise AssertionError(
                'You must provide valid lists of matching length for controlling_question_ids, choices, operators, conjunctions and locators parameters')

        for controlling_question_id in controlling_question_ids:
            self._validate_question_id(controlling_question_id)

        display_logic = {
            '0': {},
            'Type': 'BooleanExpression',
            'inPage': False
        }

        for index, controlling_question_id in enumerate(controlling_question_ids):
            locator = locators[index]
            choice = choices[index]
            operator = operators[index]
            conjunction = conjunctions[index]

            if locator:
                locator_string = 'q://{controlling_question_id}/{locator}/{choice}'.format(
                    controlling_question_id=controlling_question_id,
                    locator=locator,
                    choice=choice
                )
                conditional = 'is Selected'
            else:
                locator_string = 'q://{controlling_question_id}/{choice}'.format(
                    controlling_question_id=controlling_question_id,
                    choice=choice
                )
                conditional = 'is True'

            description = 'If {controlling_question_id} {choice} {conditional}'.format(
                controlling_question_id=controlling_question_id,
                choice=choice,
                conditional=conditional
            )

            condition = {
                'ChoiceLocator': locator_string,
                'Description': description,
                'LeftOperand': locator_string,
                'LogicType': 'Question',
                'Operator': operator,
                'QuestionID': controlling_question_id,
                "QuestionIDFromLocator": controlling_question_id,
                "QuestionIsInLoop": "no",
                'Type': 'Expression'
            }

            if operator == 'EqualTo':
                condition['RightOperand'] = '1'

            if index > 0:
                condition['Conjuction'] = conjunction

            display_logic['0'][str(index)] = condition

        display_logic['0']['Type'] = 'If'

        return display_logic

    def update_block(self, survey_id, block_id, block_description, randomization=None, block_type='Standard'):
        """

        :param survey_id:
        :param block_id:
        :param block_description:
        :param randomization: if not none expect object of form {'n': int, 'ids': list}
        :param block_type:
        :return:
        """
        try:
            assert survey_id.strip()
            assert block_id.strip()
            assert block_description.strip()
            assert block_type.strip() in self.QUALTRICS_API_SUPPORTED_BLOCK_TYPES
        except (AssertionError, AttributeError):
            raise AssertionError('You must provide string values for survey_id and block_id')

        self._validate_survey_id(survey_id)
        self._validate_block_id(block_id)

        body = {
            'Type': block_type,
            'Description': block_description,
            'Options': {
                'BlockLocking': 'false',
                'RandomizeQuestions': 'false',
                'BlockVisibility': self.QUALTRICS_API_BLOCK_VISIBILITY_COLLAPSED
            },
        }
        if randomization:
            # body['additionalProperties']: True
            body['Options']['RandomizeQuestions'] = 'Advanced'
            body['Options']['Randomization'] = {'Advanced': dict(FixedOrder=["{~SubSet~}"] * randomization['n'],
                                                                 RandomizeAll=[],
                                                                 RandomSubSet=[q_id for q_id in randomization['ids']],
                                                                 Undisplayed=[], TotalRandSubset=randomization['n'],
                                                                 QuestionsPerPage=0), 'EvenPresentation': True}
            body['BlockElements'] = [{'Type': 'Question', 'QuestionID': q_id} for q_id in randomization['ids']]

        url = '{0}/{1}/{2}/{3}/{4}'.format(
            self.base_api_url,
            self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS,
            survey_id,
            self.QUALTRICS_API_PATH_BLOCKS,
            block_id,
        )

        response = requests.put(
            url,
            data=json.dumps(body),
            headers=self._build_headers('PUT')
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise AssertionError(response.json())

    def delete_block(self, survey_id, block_id):
        try:
            assert survey_id.strip()
            assert block_id.strip()
        except (AssertionError, AttributeError):
            raise AssertionError('You must provide string values for survey_id and block_id')

        self._validate_survey_id(survey_id)
        self._validate_block_id(block_id)

        url = '{0}/{1}/{2}/{3}/{4}'.format(
            self.base_api_url,
            self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS,
            survey_id,
            self.QUALTRICS_API_PATH_BLOCKS,
            block_id,
        )

        response = requests.delete(
            url,
            headers=self._build_headers('DELETE')
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise AssertionError(response.json())

    def create_block(self, survey_id=TEST_SURVEY_ID, block_name='Default Block', block_type='Standard'):
        try:
            assert survey_id.strip()
            assert block_name.strip()
        except (AssertionError, AttributeError):
            raise AssertionError('You must provide string values for survey_id and description')

        assert block_type in self.QUALTRICS_API_SUPPORTED_BLOCK_TYPES, 'Supplied block type is not supported.'

        self._validate_survey_id(survey_id)

        url = '{0}/{1}/{2}/{3}'.format(
            self.base_api_url,
            self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS,
            survey_id,
            self.QUALTRICS_API_PATH_BLOCKS
        )

        body = json.dumps({
            'Type': block_type,
            'Description': block_name,
            'Options': {
                'BlockLocking': 'false',
                'RandomizeQuestions': 'false',
                'BlockVisibility': self.QUALTRICS_API_BLOCK_VISIBILITY_COLLAPSED
            }
        })

        response = requests.post(
            url,
            data=body,
            headers=self._build_headers('POST')
        )

        response.raise_for_status()

        result = response.json()

        block_id = result['result']['BlockID']

        return {
            'response': result,
            'block_id': block_id
        }

    def _update_survey(self, survey_id, questions_per_page=QUESTIONS_PER_PAGE):  # , is_active):
        try:
            assert survey_id.strip()
            # assert isinstance(is_active, bool)
        except (AssertionError, AttributeError):
            # raise AssertionError('You must provide a string value for survey_id and a bool for is_active')
            raise AssertionError('No survey ID given')

        self._validate_survey_id(survey_id)

        body = json.dumps({
            # 'isActive': is_active
            'QuestionsPerPage': '{}'.format(questions_per_page),
            'BackButton': False,
            'BallotBoxStuffingPrevention': False,
            'EOSRedirectURL': EOSRedirectURL,
            'ProgressBarDisplay': 'Text',
            'Header': '',
            'Footer': '',
            'NoIndex': 'Yes',
            'NextButton': ' → ',
            'PartialData': '+1 week',
            'PreviousButton': ' ← ',
            'SaveAndContinue': True,
            'SecureResponseFiles': "true",
            "SurveyTermination": "Redirect",
            "SurveyProtection": "PublicSurvey",
            "RecaptchaV3": "true",
            "SurveyExpiration": 'None',
        })
        # "ValidationMessage": 'null',
        # "SurveyExpirationDate": "2021-07-01 00:00:00",

        url = '{0}/{1}/{2}/{3}'.format(
            self.base_api_url,
            self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS,
            survey_id,
            self.QUALTRICS_API_PATH_SURVEY_OPTIONS,
        )

        response = requests.put(
            url,
            data=body,
            headers=self._build_headers('PUT')
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            logging.info(response.json())
            raise Exception('Error with HTTP: {}'.format(response.content))

    def create_survey(self, survey_name: str, language_code='EN', project_category='CORE'):
        url = '{0}/{1}'.format(self.base_api_url, self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS)

        body = json.dumps({
            'SurveyName': survey_name,
            'Language': language_code.upper(),
            'ProjectCategory': project_category.upper(),
        })

        response = requests.post(
            url,
            data=body,
            headers=self._build_headers('POST')
        )

        response.raise_for_status()

        result = response.json()

        survey_id = result['result']['SurveyID']
        logging.info('survey id is {}'.format(survey_id))
        default_block_id = result['result']['DefaultBlockID']

        return survey_id, default_block_id

    def delete_survey(self, survey_id: str, language_code='EN', project_category='CORE', survey_name='my-api-test'):
        # TODO: why is this an error?!
        #  According to this: https://api.qualtrics.com/api-reference/reference/surveyDefinitions.json/paths/~1survey-definitions~1%7BsurveyId%7D/delete
        #  it should work
        url = '{0}/{1}/{2}'.format(self.base_api_url, self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS, survey_id)
        body = json.dumps({
            'SurveyName': survey_name,
            'Language': language_code.upper(),
            'ProjectCategory': project_category.upper(),
        })
        response = requests.post(
            url,
            data=body,
            # headers=self._build_headers('GET')
        )
        response.raise_for_status()
        logging.info(response.json())

    def get_survey(self, survey_id: str):
        try:
            assert survey_id.strip()
        except (AssertionError, AttributeError):
            raise AssertionError('You must provide a string value for survey_id')

        self._validate_survey_id(survey_id)

        url = '{0}/{1}/{2}'.format(
            self.base_api_url,
            self.QUALTRICS_API_PATH_SURVEYS,
            survey_id,
        )

        response = requests.get(
            url,
            headers=self._build_headers('GET')
        )

        response.raise_for_status()

        return response.json()

    def create_triple_survey(self, survey_name: str = ''):
        # survey_id, default_block_id = self.create_survey(survey_name)
        survey_id = "SV_domMYEvDr9Aq7qe"
        # only display one question per page
        self._update_survey(survey_id, questions_per_page=1)

        # add welcome block
        import qualtrics_constants
        result, block_id = self.create_block(survey_id=survey_id, block_name='Welcome Message')
        self.add_text_block(survey_id=survey_id, block_id=block_id, text=qualtrics_constants.triple_task_description,
                            text_id="TB_welcome")
        # add example block
        for triple_intro_ex in qualtrics_constants.triple_intro_exs:
            self.add_triple_q(survey_id=survey_id,
                              anchor_text=triple_intro_ex["anchor"], anchor_id=triple_intro_ex["anchor_id"],
                              alternative_0_text=triple_intro_ex["alternative_0"],
                              alternative_0_id=triple_intro_ex["alternative_0_id"],
                              alternative_1_text=triple_intro_ex["alternative_1"],
                              alternative_1_id=triple_intro_ex["alternative_1_id"],
                              block_id=block_id, correct_alternative=1,
                              correct_msg=triple_intro_ex["correct_msg"],
                              wrong_msg=triple_intro_ex["wrong_msg"])
        # add end example block
        self.add_text_block(survey_id=survey_id, block_id=block_id, text=qualtrics_constants.triple_end_intro_ex_msg,
                            text_id="TB_welcome_end")
        # add triple block
        # add test block
        # add end block

    def add_triple_q(self, anchor_text, anchor_id,
                     alternative_0_text, alternative_0_id,
                     alternative_1_text, alternative_1_id,
                     block_id, correct_alternative: int,
                     survey_id=TEST_SURVEY_ID, correct_msg=None,
                     wrong_msg=None):  # scoring_id=,
        # CREATE TRIPLE QUESTION
        triple_q_id = "QT_{0}_{1}_{2}--{3}".format(anchor_id, alternative_0_id, alternative_1_id,
                                                   correct_alternative)
        question_params = {
            "text": "Given the text snippet<br /><br />&nbsp;&nbsp; <strong>&quot;{}&quot;</strong>,<br /><br />"
                    "which of the following is more consistent in linguistic style?".format(anchor_text),
            "tag_number": triple_q_id,
            "type": MULTIPLE_CHOICE_TYPE,
            "translations": [],
            "answer_selector": MULTIPLE_CHOICE_SELECTOR,
            "label": "Triple Question {}".format(triple_q_id),
            "choices": {
                "1": {
                    "Display": "{}".format(alternative_0_text)
                },
                "2": {
                    "Display": "{}".format(alternative_1_text)
                }
            },
            "choice_order": ["1", "2"],
            'answer_sub_selector': 'TX',
            'is_mandatory': True,
            'block_number': 0,
        }
        question_payload = self._build_question_payload(question_params, survey_id=survey_id)
        result, question_id = self._create_question(survey_id, question_payload, block_id=block_id)

        return {
            "qualtric_q_id": question_id,
            "question_tag": triple_q_id
        }

    def _build_question_payload(self, question_params=None, survey_id=None,
                                include_display_logic=False):  # pylint: disable=too-many-branches, too-many-statements
        try:
            assert question_params

            keys_present = True
            for required_key in self._qualtrics_api_required_question_param_keys:
                if required_key not in question_params.keys():
                    keys_present = False

            assert keys_present

            assert question_params['text'].strip()
            # assert isinstance(question_params['tag_number'], int)
            assert question_params['type'].strip() in self._qualtrics_api_supported_question_types
            assert question_params['answer_selector'].strip() in self._qualtrics_api_supported_answer_selectors
            assert question_params['label'].strip()
            assert isinstance(question_params['block_number'], int)
        except (KeyError, AssertionError, AttributeError):
            raise AssertionError('Please ensure the question_params dictionary argument is valid...')

        # Constructing question payload dictionary
        payload = {
            'QuestionText': question_params['text'].strip(),
            'DataExportTag': '{}'.format(question_params['tag_number']),
            'QuestionID': '{}'.format(question_params['tag_number']),
            'QuestionType': question_params['type'],
            'Selector': question_params['answer_selector'],
            'Configuration': {
                'QuestionDescriptionOption': 'SpecifyLabel'
            },
            'QuestionDescription': question_params['label'].strip(),
            'Validation': {
                'Settings': {
                    'ForceResponse': 'OFF' if not question_params['is_mandatory'] else 'ON',
                    'ForceResponseType': 'ON',
                    'Type': 'None'
                }
            },
            'Language': question_params['translations']
        }

        if 'answer_sub_selector' in question_params.keys():
            payload['SubSelector'] = question_params['answer_sub_selector']

        if 'additional_validation_settings' in question_params.keys():
            for key, value in question_params['additional_validation_settings'].items():
                payload['Validation']['Settings'][key] = value

        # Applying choices, order, recoding and variable naming to payload dictionary

        if 'choices' in question_params.keys():
            payload['Choices'] = question_params['choices']

        if 'groups' in question_params.keys():
            payload['Groups'] = question_params['groups']
            payload['NumberOfGroups'] = 1

        if 'choice_order' in question_params.keys():
            payload['ChoiceOrder'] = question_params['choice_order']

        if 'recode_values' in question_params.keys():
            payload['RecodeValues'] = question_params['recode_values']

        if 'variable_naming' in question_params.keys():
            payload['VariableNaming'] = question_params['variable_naming']

        if 'column_labels' in question_params.keys():
            payload['ColumnLabels'] = question_params['column_labels']

        if 'grading_data' in question_params.keys():
            payload['GradingData'] = question_params['grading_data']

        if 'default_choices' in question_params.keys():
            payload['DefaultChoices'] = question_params['default_choices']

        # Processing question display_logic
        if include_display_logic:
            if 'display_logic' in question_params.keys():
                question_display_logic_dict = question_params['display_logic']

                if question_display_logic_dict:
                    # unpack dict entries for use as params below
                    controlling_question_labels = question_display_logic_dict['controlling_question_labels']
                    choices = question_display_logic_dict['choices']
                    operators = question_display_logic_dict['operators']
                    conjunctions = question_display_logic_dict['conjunctions']
                    locators = question_display_logic_dict['locators']

                    # the QIDn of this question could vary, so get this from the survey by the question label
                    controlling_question_ids = []
                    for controlling_question_label in controlling_question_labels:
                        controlling_question_id, _ = self._find_question_in_survey_by_label(survey_id,
                                                                                            controlling_question_label)
                        controlling_question_ids.append(controlling_question_id)

                    # finally add a 'DisplayLogic' key with a value of the built display logic to this question payload
                    payload['DisplayLogic'] = self._build_question_display_logic(controlling_question_ids, choices,
                                                                                 operators, conjunctions, locators)

        return payload

    def _create_question(self, survey_id: str, question_payload, block_id=None):
        # https://github.com/60decibels/pytrics/blob/91ab8107433be0391f0fad06fa702eaa93f6af21/pytrics/qualtrics_api/client.py#L251
        try:
            assert survey_id.strip()
            assert question_payload
            assert isinstance(question_payload, dict)
            if block_id:
                assert block_id.strip()
        except (AssertionError, AttributeError):
            raise AssertionError('You must provide a string value for survey_id and a dict for question_data')

        self._validate_survey_id(survey_id)
        self._validate_question_payload(question_payload)
        if block_id:
            self._validate_block_id(block_id)

        url = '{0}/{1}/{2}/{3}'.format(
            self.base_api_url,
            self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS,
            survey_id,
            self.qualtrics_api_path_questions
        )

        if block_id:
            url = '{0}?blockId={1}'.format(url, block_id)

        body = json.dumps(question_payload)

        response = requests.post(
            url,
            data=body,
            headers=self._build_headers('POST')
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            logging.info(response.json())
            raise Exception('Error with HTTP: {}'.format(response.content))

        result = response.json()

        question_id = result['result']['QuestionID']

        return result, question_id


class TripQuadSurveyBuilder(QualtricsAPI):

    def __init__(self):
        super(TripQuadSurveyBuilder, self).__init__()

    def add_welcome_block(self, survey_id=TEST_SURVEY_ID):
        block_id = self.create_block(survey_id, 'Welcome Block')['block_id']
        # ADD text entry question for Prolific ID
        question_params = self._build_question_params(qualtrics_constants.prolific_msg, q_id=QID_PROLIFIC_PID,
                                                      q_type=TEXT_ENTRY_TYPE, answer_selector=TEXT_ENTRY_SELECTOR,
                                                      label="Prolific ID question", forced_response=True,
                                                      default_choice=True)
        question_payload = self._build_question_payload(question_params, survey_id)
        self._create_question(survey_id, question_payload, block_id)
        # ADD multiple choice question for consent
        question_params = self._build_question_params(qualtrics_constants.consent_msg, q_id='Q-welcome_consent',
                                                      q_type=MULTIPLE_CHOICE_TYPE,
                                                      answer_selector=MULTIPLE_CHOICE_SELECTOR,
                                                      label="Consent question", forced_response=True,
                                                      choices=qualtrics_constants.consent_choices,
                                                      answer_sub_selector=MULTIPLE_CHOICE_SUBSELECTOR)
        question_payload = self._build_question_payload(question_params, survey_id)
        _, consent_id = self._create_question(survey_id, question_payload, block_id)
        return block_id, consent_id

    def add_task_description_block(self, survey_id=TEST_SURVEY_ID, triple=True):
        if triple:
            block_id = self.create_block(survey_id=survey_id, block_name='Triple Task Description')['block_id']
            task_description = qualtrics_constants.triple_task_description
        else:
            block_id = self.create_block(survey_id=survey_id, block_name='Quadruple Task Description')['block_id']
            task_description = qualtrics_constants.quadruple_task_description
        text_params = self._build_question_params(task_description, q_id='Q-Text_Description',
                                                  q_type=TEXT_ONLY_TYPE, answer_selector=TEXT_ONLY_SELECTOR,
                                                  label='Task Description')
        self._create_question(survey_id=survey_id,
                              question_payload=self._build_question_payload(text_params, survey_id),
                              block_id=block_id)

        return block_id

    def add_end_block(self, survey_id=TEST_SURVEY_ID):
        block_id = self.create_block(survey_id=survey_id, block_name='End Block')['block_id']
        logging.info("ADDING end block with id {}".format(block_id))
        question_params = self._build_question_params(qualtrics_constants.end_comment_msg, q_id='Q-end_comment',
                                                      q_type=TEXT_ENTRY_TYPE, answer_selector=TEXT_ENTRY_SELECTOR,
                                                      label="End comment question", forced_response=False)
        self._create_question(survey_id, self._build_question_payload(question_params, survey_id), block_id)
        return block_id

    def add_screen_block(self, survey_id=TEST_SURVEY_ID, triple=True, screen_nbr=1):
        block_name = 'Screening Block {}'.format(screen_nbr)
        block_id = self.create_block(survey_id, block_name)['block_id']
        logging.info("ADDING screen block {} with id {}".format(screen_nbr, block_id))

        import pandas as pd
        df_screen_qs = pd.read_csv(SCREEN_QS_FILENAME, sep='\t')

        qualtrics_ids = []
        q_tags = []

        for i, q in df_screen_qs.iterrows():
            if triple:
                qualtrics_id, q_tag = self._create_triple_q_from_df_row(q, block_id, survey_id)
            else:
                qualtrics_id, q_tag = self._create_quad_q_from_df_row(q, block_id, survey_id)
            qualtrics_ids.append(qualtrics_id)
            q_tags.append(q_tag)
        return {
            'block_id': block_id,
            'block_name': block_name,
            'qualtrics_ids': qualtrics_ids,
            'q_ids': q_tags,
            'df_screen_qs': df_screen_qs
        }

    def add_main_q_block(self, survey_id=TEST_SURVEY_ID, df_filename=QUAD_QS_FILENAME, limit_q=None, triple=True,
                         added_qs=None, block_nbr=1):
        '''
        adds a main question block & a dummy question block with the identical questions
        :param added_qs: already added qs in previous calls signified with their question IDs,
            NONE if there have none been added
        :param survey_id: which survey to add to
        :param df_filename: filename with the questions to sample from
        :param limit_q: number of questions to add
        :param triple: boolean whether to add questions in triple format
        :return:
        '''
        keyword = TRIPLE
        if not triple:
            keyword = QUADRUPLE
        block_name = '{} Questions Block {}'.format(keyword, block_nbr)
        block_id = self.create_block(survey_id, block_name)['block_id']
        logging.info("ADDING main {} question block number {} with id {}".format(keyword, block_nbr, block_id))
        d_block_name = 'Dummy {} Questions Block {}'.format(keyword, block_nbr)
        d_block_id = self.create_block(survey_id, d_block_name)['block_id']
        logging.info("ADDING main {} dummy question block with id {}".format(keyword, d_block_id))
        logging.info("       ADDING questions from {}".format(df_filename))

        quad_df = pd.read_csv(df_filename, sep='\t')
        sampled_qs = quad_df

        if limit_q:
            nbr_q_per_dim = int(limit_q / 2)
            logging.info("ADDING {0} questions with equal share of {1} formal/informal and {1} simple/complex questions"
                         .format(limit_q, nbr_q_per_dim))

            pot_qs = quad_df[quad_df[IN_SUBSAMPLE_COL] == True]
            if limit_q > len(pot_qs):
                raise AttributeError("can not extract {} from a sample of {} questions".format(limit_q, len(pot_qs)))

            if added_qs is not None:
                # only sample from questions not in added_qs
                pot_qs = quad_df[~quad_df[ID_COL].isin(added_qs)]

            if limit_q <= len(pot_qs):
                # sampled_qs = pd.DataFrame().reindex_like(pot_qs).dropna()
                q_indices = []
                # formal
                for dim_tag in ['c-', 'f-']:
                    i = 0
                    while i < nbr_q_per_dim:  # TODO: use sample n=limitq not loop
                        pot_q = pot_qs.sample(1)
                        q_tag = pot_q[set_for_global.ID_COL].values[0]
                        if dim_tag in q_tag and q_tag not in q_indices:
                            q_indices.append(q_tag)
                            i += 1
                sampled_qs = pot_qs[pot_qs[set_for_global.ID_COL].isin(q_indices)]
            else:
                sampled_qs = pot_qs

            d_q_ids, nbr_qs_added, q_ids = self.add_qs_to_survey(block_id, d_block_id, sampled_qs, survey_id, triple)
        else:
            d_q_ids, nbr_qs_added, q_ids = self.add_qs_to_survey(block_id, d_block_id, quad_df, survey_id, triple)

        logging.info("ADDED {} main and dummy questions respectively".format(nbr_qs_added))
        return {
            'block_id': block_id,
            'block_name': block_name,
            'qualtrics_ids': q_ids,
            'd_block_id': d_block_id,
            'd_block_name': d_block_name,
            'd_qualtrics_ids': d_q_ids,
            'q_tags': sampled_qs,
            'quad_df': quad_df
        }

    def add_qs_to_survey(self, block_id, d_block_id, quad_q_to_add, survey_id, triple):
        q_ids = []
        d_q_ids = []
        for i, question in quad_q_to_add.iterrows():
            if triple:
                q_id, _ = self._create_triple_q_from_df_row(question, block_id, survey_id)
                d_q_id, _ = self._create_triple_q_from_df_row(question, d_block_id, survey_id, dummy=True)
            else:
                q_id, _ = self._create_quad_q_from_df_row(question, block_id, survey_id)
                d_q_id, _ = self._create_quad_q_from_df_row(question, d_block_id, survey_id, dummy=True)
            q_ids.append(q_id)
            d_q_ids.append(d_q_id)
        return d_q_ids, len(quad_q_to_add), q_ids

    def set_meta_attributes(self, survey_id=TEST_SURVEY_ID):
        # only display one question per page
        self._update_survey(survey_id, questions_per_page=1)

    def create_triple_survey(self, survey_name: str = 'API-Triple-Pilot', limit_q=None, q_to_display=14,
                             nbr_screen_qs=2, df_filename=QUAD_QS_FILENAME):
        self.create_triple_quad_survey(survey_name, limit_q=limit_q, triple=True, q_to_display=q_to_display,
                                       nbr_screen_qs=nbr_screen_qs, df_filename=df_filename)

    def randomize_block(self, survey_id, block_id, block_name, qualtrics_ids, nbr_to_display=14, normal_rand=False):
        logging.info("RANDOMIZING {}".format(block_name))
        assert nbr_to_display <= len(qualtrics_ids), 'Too few questions presented ... '
        randomization = {
            'n': nbr_to_display,
            'ids': qualtrics_ids
        }
        self.update_block(survey_id, block_id, block_name, randomization=randomization)

    def create_triple_quad_survey(self, survey_name, limit_q=None, triple=True, q_to_display=14, nbr_screen_qs=2,
                                  nbr_blocks=1, df_filename=QUAD_QS_FILENAME):
        set_for_global.set_global_seed(w_torch=False)
        survey_id, default_block = self.create_survey(survey_name)
        self._update_survey(survey_id, questions_per_page=1)
        welcome_id, consent_id = self.add_welcome_block(survey_id=survey_id)

        description_id = self.add_task_description_block(survey_id=survey_id, triple=triple)
        screen_dict = self.add_screen_block(survey_id=survey_id, triple=triple)
        # ADD main question block
        main_block_dict = self.add_main_q_block(survey_id, limit_q=limit_q, triple=triple, df_filename=df_filename)
        self.randomize_block(survey_id, main_block_dict['block_id'], main_block_dict['block_name'],
                             main_block_dict['qualtrics_ids'], nbr_to_display=q_to_display)
        self.randomize_block(survey_id, main_block_dict['d_block_id'], main_block_dict['d_block_name'],
                             main_block_dict['d_qualtrics_ids'], nbr_to_display=q_to_display)
        self.randomize_block(survey_id, screen_dict['block_id'], screen_dict['block_name'],
                             screen_dict['qualtrics_ids'], nbr_to_display=nbr_screen_qs, normal_rand=True)

        end_id = self.add_end_block(survey_id)

        self.add_branch_logic(survey_id, screen_dict['qualtrics_ids'], screen_dict['q_ids'],
                              main_block_dict['d_block_id'], welcome_id, consent_id, description_id,
                              screen_dict['block_id'], main_block_dict['block_id'], end_id,
                              screen_dict['df_screen_qs'], triple=triple)

    def add_branch_logic(self, survey_id, qualtrics_question_ids, q_ids, dummy_block_id, welcome_block_id, consent_id,
                         description_id, screen_id, main_id, end_id, screen_df, triple=False):

        logging.info("ADDING branch logic ...")
        url = '{0}/{1}/{2}/{3}'.format(
            self.base_api_url,
            self.QUALTRICS_API_PATH_SURVEY_DEFINITIONS,
            survey_id,
            self.QUALTRICS_API_PATH_FLOW
        )

        screen_branch_logic = self.get_screen_branch_logic(q_ids, qualtrics_question_ids, screen_df, triple)

        body = {
            "Type": "Root",
            "FlowID": "FL_1",
            "Flow": [
                {
                    "Type": "EmbeddedData",
                    "FlowID": "FL_prolific",
                    "EmbeddedData": [
                        {
                            "Description": "PROLIFIC_PID",
                            "Type": "Recipient",
                            "Field": "PROLIFIC_PID",
                            "VariableType": "Nominal",
                            "DataVisibility": []
                        },
                        {
                            "Description": "Q_Response_Type",
                            "Type": "Custom",
                            "Field": "Q_Response_Type",
                            "VariableType": "MultiValueNominal",
                            "DataVisibility": [],
                            "Value": "In Process"
                        }
                    ]
                },
                {
                    "Type": "Standard",
                    "ID": welcome_block_id,
                    "FlowID": "FL_4",
                    "Autofill": []
                },
                {
                    "Type": "Branch",
                    "FlowID": "FL_10",
                    "Description": "New Branch",
                    "BranchLogic": {
                        "0": {
                            "0": {
                                "LogicType": "Question",
                                "QuestionID": "QID2",
                                "QuestionIsInLoop": "no",
                                "ChoiceLocator": "q://{}/SelectableChoice/1".format(consent_id),
                                "Operator": "NotSelected",
                                "QuestionIDFromLocator": "{}".format(consent_id),
                                "LeftOperand": "q://{}/SelectableChoice/1".format(consent_id),
                                "Type": "Expression",
                                "Description": "<span class=\"ConjDesc\">If</span> <span class=\"QuestionDesc\">Consent question</span> <span class=\"LeftOpDesc\">Yes, begin with the study.</span> <span class=\"OpDesc\">Is Not Selected</span> "
                            },
                            "Type": "If"
                        },
                        "Type": "BooleanExpression"
                    },
                    "Flow": [
                        {
                            "Type": "EndSurvey",
                            "FlowID": "FL_11",
                            "EndingType": "Advanced",
                            "Options": {
                                "Advanced": "true",
                                "SurveyTermination": "DisplayMessage",
                                "EOSMessageLibrary": "UR_9QSK4XhrWgHS1ZY",
                                "EOSMessage": "MS_7Og8PlKhCteovBk"
                            }
                        }
                    ]
                },
                {
                    "Type": "Standard",
                    "ID": description_id,
                    "FlowID": "FL_5",
                    "Autofill": []
                },
                {
                    "Type": "Standard",
                    "ID": screen_id,
                    "FlowID": "FL_6",
                    "Autofill": []
                },
                {
                    "Type": "Branch",
                    "FlowID": "FL_12",
                    "Description": "New Branch",
                    "BranchLogic": {
                        "0": screen_branch_logic,
                        "Type": "BooleanExpression"
                    },
                    "Flow": [
                        {
                            "Type": "Standard",
                            "ID": dummy_block_id,
                            "FlowID": "FL_8",
                            "Autofill": []
                        },
                        {
                            "Type": "Standard",
                            "ID": end_id,
                            "FlowID": "FL_18",
                            "Autofill": []
                        },
                        {
                            "Type": "EmbeddedData",
                            "FlowID": "FL_17",
                            "EmbeddedData": [
                                {
                                    "Description": "Q_Response_Type",
                                    "Type": "Custom",
                                    "Field": "Q_Response_Type",
                                    "VariableType": "MultiValueNominal",
                                    "DataVisibility": [],
                                    "Value": "Screened Out"
                                }
                            ]
                        },
                        {
                            "Type": "EndSurvey",
                            "FlowID": "FL_13",
                            "EndingType": "Advanced",
                            "Options": {
                                "Advanced": "true",
                                "SurveyTermination": "Redirect",
                                "EOSRedirectURL": EOSRedirectURL,
                                "ResponseFlag": "Screened"
                            }
                        }
                    ]
                },
                {
                    "Type": "Standard",
                    "ID": main_id,
                    "FlowID": "FL_7",
                    "Autofill": []
                },
                {
                    "Type": "Standard",
                    "ID": end_id,
                    "FlowID": "FL_9",
                    "Autofill": []
                },
                {
                    "Type": "EmbeddedData",
                    "FlowID": "FL_16",
                    "EmbeddedData": [
                        {
                            "Description": "Q_Response_Type",
                            "Type": "Custom",
                            "Field": "Q_Response_Type",
                            "VariableType": "MultiValueNominal",
                            "DataVisibility": [],
                            "Value": "Valid"
                        }
                    ]
                }
            ],
            "Properties": {
                "Count": 18
            }
        }

        response = requests.put(
            url,
            data=json.dumps(body),
            headers=self._build_headers('PUT')
        )
        logging.info(response.json())
        response.raise_for_status()

        return response.json()

    def get_screen_branch_logic(self, q_ids, qualtrics_question_ids, screen_df, triple):
        if not triple:
            screen_branch_logic = {i: dict(LogicType="Question", QuestionID=qualtrics_question_ids[i],
                                           QuestionIsInLoop="no",
                                           ChoiceLocator="q://{}/ChoiceNumericEntryValue/Rank/1"
                                           .format(qualtrics_question_ids[i]), Operator="EqualTo",
                                           QuestionIDFromLocator="{}".format(qualtrics_question_ids[i]),
                                           LeftOperand="q://{}/ChoiceNumericEntryValue/Rank/1"
                                           .format(qualtrics_question_ids[i]),
                                           RightOperand='2' if int(q_ids[i].split('--')[1]) == 0 else '1',
                                           Type="Expression",
                                           Description="<span class=\"ConjDesc\">Or</span> <span class=\"QuestionDesc\">{}"
                                                       "</span> <span class=\"LeftOpDesc\">{}</span> "
                                                       "<span class=\"OpDesc\">Is Equal to</span> "
                                                       "<span class=\"RightOpDesc\"> {} </span>"
                                           .format(q_ids[i],
                                                   screen_df[screen_df[ID_COL] == q_ids[i]][ALTERNATIVE11_COL].values[
                                                       0],
                                                   '2' if int(q_ids[i].split('--')[1]) == 0 else '1'))
                                   for i in range(len(qualtrics_question_ids))}
            screen_branch_logic[0]["Description"] = "<span class=\"ConjDesc\">If</span> <span class=\"QuestionDesc\">{}" \
                                                    "</span> <span class=\"LeftOpDesc\">{}</span> " \
                                                    "<span class=\"OpDesc\">Is Equal to</span> " \
                                                    "<span class=\"RightOpDesc\"> {} </span>".format(q_ids[0],
                                                                                                     screen_df[
                                                                                                         screen_df[
                                                                                                             ID_COL] ==
                                                                                                         q_ids[0]][
                                                                                                         ALTERNATIVE11_COL].values[
                                                                                                         0],
                                                                                                     '2' if int(
                                                                                                         q_ids[0].split(
                                                                                                             '--')[
                                                                                                             1]) == 0 else '1')
        else:
            q_ids = [tag.replace('QT-', 'QQ-') for tag in q_ids]
            screen_branch_logic = {i: dict(LogicType="Question", QuestionID=qualtrics_question_ids[i],
                                           QuestionIsInLoop="no",
                                           ChoiceLocator="q://{}/SelectableChoice/{}"
                                           .format(qualtrics_question_ids[i],
                                                   '2' if int(q_ids[i].split('--')[1]) == 0 else '1'),
                                           Operator="Selected",
                                           QuestionIDFromLocator="{}".format(qualtrics_question_ids[i]),
                                           LeftOperand="q://{}/SelectableChoice/{}"
                                           .format(qualtrics_question_ids[i],
                                                   '2' if int(q_ids[i].split('--')[1]) == 0 else '1'),
                                           Type="Expression",
                                           Description="<span class=\"ConjDesc\">Or</span> <span class=\"QuestionDesc\">{}"
                                                       "</span> <span class=\"LeftOpDesc\">{}</span> "
                                                       "<span class=\"OpDesc\">Is Selected</span> "
                                           .format(q_ids[i].replace('QQ-', 'QT-'),
                                                   screen_df[screen_df[ID_COL] == q_ids[i]][ALTERNATIVE11_COL].values[
                                                       0],
                                                   '2' if int(q_ids[i].split('--')[1]) == 0 else '1'))
                                   for i in range(len(qualtrics_question_ids))}
            screen_branch_logic[0]["Description"] = "<span class=\"ConjDesc\">Or</span> <span class=\"QuestionDesc\">{}" \
                                                    "</span> <span class=\"LeftOpDesc\">{}</span> " \
                                                    "<span class=\"OpDesc\">Is Selected</span> " \
                .format(q_ids[0].replace('QQ-', 'QT-'),
                        screen_df[screen_df[ID_COL] == q_ids[0]][ALTERNATIVE11_COL].values[0],
                        '2' if int(q_ids[0].split('--')[1]) == 0 else '1')
        for i in range(len(qualtrics_question_ids) - 1):
            screen_branch_logic[i + 1][BRANCH_CONJUCTION] = "Or"
        screen_branch_logic["Type"] = "If"
        return screen_branch_logic

    def get_survey_end(self, end_id, screening_flow):
        dummy_end_survey_dict = {
            "Type": "EndSurvey",
            "FlowID": "FL_end_",
            "EndingType": "Advanced",
            "Options": {
                "Advanced": "true",
                "SurveyTermination": "Redirect",
                "EOSRedirectURL": EOSRedirectURL,
                "ResponseFlag": "Screened"
            }
        }
        screening_flow[1]["Flow"].append(dummy_end_survey_dict)
        end_dict_list = [
            {
                "Type": "Standard",
                "ID": end_id,
                "FlowID": "FL_9",
                "Autofill": []
            },
            {
                "Type": "EmbeddedData",
                "FlowID": "FL_valid",
                "EmbeddedData": [
                    {
                        "Description": RESPONSE_TYPE_COL,
                        "Type": "Custom",
                        "Field": RESPONSE_TYPE_COL,
                        "VariableType": "MultiValueNominal",
                        "DataVisibility": [],
                        "Value": VALID_RESPONSE
                    }
                ]
            }
        ]
        return end_dict_list

    def get_screen_branching(self, block_dict_by_nbr, main_block_nbr, screen_branch_logic, screen_dict_by_nbr):
        screening_flow = [{
            "Type": "Standard",
            "ID": screen_dict_by_nbr[main_block_nbr]['block_id'],
            "FlowID": "FL_screen_{}".format(main_block_nbr),
            "Autofill": []
        }, {
            "Type": "Branch",
            "FlowID": "FL_screencheck_{}".format(main_block_nbr),
            "Description": "New Branch",
            "BranchLogic": {
                "0": screen_branch_logic,
                "Type": "BooleanExpression"
            },
            "Flow": [
                {
                    "Type": "Standard",
                    "ID": block_dict_by_nbr[main_block_nbr]['d_block_id'],
                    "FlowID": "FL_dummy_{}".format(main_block_nbr),
                    "Autofill": []
                },
                {
                    "Type": "EmbeddedData",
                    "FlowID": "FL_embed_{}".format(main_block_nbr),
                    "EmbeddedData": [
                        {
                            "Description": RESPONSE_TYPE_COL,
                            "Type": "Custom",
                            "Field": RESPONSE_TYPE_COL,
                            "VariableType": "MultiValueNominal",
                            "DataVisibility": [],
                            "Value": EMBED_SCREENED_OUT
                        }
                    ]
                },
            ]}, {
            "Type": "Standard",
            "ID": block_dict_by_nbr[main_block_nbr]['block_id'],
            "FlowID": "FL_mainblock_{}".format(main_block_nbr),
            "Autofill": []
        }]
        return screening_flow

    def get_begin_flow(self, consent_id, description_id, welcome_block_id):
        base_body = {
            "Type": "Root",
            "FlowID": "FL_root",
            "Flow": [
                {
                    "Type": "EmbeddedData",
                    "FlowID": "FL_prolific",
                    "EmbeddedData": [
                        {
                            "Description": "PROLIFIC_PID",
                            "Type": "Recipient",
                            "Field": "PROLIFIC_PID",
                            "VariableType": "Nominal",
                            "DataVisibility": []
                        },
                        {
                            "Description": RESPONSE_TYPE_COL,
                            "Type": "Custom",
                            "Field": RESPONSE_TYPE_COL,
                            "VariableType": "MultiValueNominal",
                            "DataVisibility": [],
                            "Value": "In Process"
                        }
                    ]
                },
                {
                    "Type": "Standard",
                    "ID": welcome_block_id,
                    "FlowID": "FL_welcome",
                    "Autofill": []
                },
                {
                    "Type": "Branch",
                    "FlowID": "FL_consentbranch",
                    "Description": "New Branch",
                    "BranchLogic": {
                        "0": {
                            "0": {
                                "LogicType": "Question",
                                "QuestionID": "QID2",
                                "QuestionIsInLoop": "no",
                                "ChoiceLocator": "q://{}/SelectableChoice/1".format(consent_id),
                                "Operator": "NotSelected",
                                "QuestionIDFromLocator": "{}".format(consent_id),
                                "LeftOperand": "q://{}/SelectableChoice/1".format(consent_id),
                                "Type": "Expression",
                                "Description": "<span class=\"ConjDesc\">If</span> <span class=\"QuestionDesc\">Consent question</span> <span class=\"LeftOpDesc\">Yes, begin with the study.</span> <span class=\"OpDesc\">Is Not Selected</span> "
                            },
                            "Type": "If"
                        },
                        "Type": "BooleanExpression"
                    },
                    "Flow": [
                        {
                            "Type": "EndSurvey",
                            "FlowID": "FL_endnoconsent",
                            "EndingType": "Advanced",
                            "Options": {
                                "Advanced": "true",
                                "SurveyTermination": "DisplayMessage",
                                "EOSMessageLibrary": "UR_9QSK4XhrWgHS1ZY",
                                "EOSMessage": "MS_7Og8PlKhCteovBk"
                            }
                        }
                    ]
                },
                {
                    "Type": "Standard",
                    "ID": description_id,
                    "FlowID": "FL_taskdescription",
                    "Autofill": []
                },
            ]}
        return base_body

    def get_screen_condition(self, main_block_nbr, screen_dict_by_nbr, triple=True):
        qualtrics_question_ids = screen_dict_by_nbr[main_block_nbr]['qualtrics_ids']
        q_ids = screen_dict_by_nbr[main_block_nbr]['q_ids']
        if triple:
            return {i: dict(LogicType="Question",
                            QuestionID=screen_dict_by_nbr[main_block_nbr]['qualtrics_ids'][i],
                            QuestionIsInLoop="no",
                            ChoiceLocator="q://{}/SelectableChoice/{}".format(
                                screen_dict_by_nbr[main_block_nbr]['qualtrics_ids'][i],
                                '2' if int(screen_dict_by_nbr[main_block_nbr]['q_ids'][i]
                                           .split('--')[1]) == 0 else '1')
                            .format(screen_dict_by_nbr[main_block_nbr]['qualtrics_ids'][i]),
                            Operator="Selected",
                            QuestionIDFromLocator="{}".format(screen_dict_by_nbr[main_block_nbr]['qualtrics_ids'][i]),
                            LeftOperand="q://{}/SelectableChoice/{}".format(
                                screen_dict_by_nbr[main_block_nbr]['qualtrics_ids'][i],
                                2 if int(screen_dict_by_nbr[main_block_nbr]['q_ids'][i]
                                           .split('--')[1]) == 0 else 1),
                            Type="Expression",
                            Description="Test for wrong answer in {}".format(
                                screen_dict_by_nbr[main_block_nbr]['q_ids'][i]),
                            Conjunction="Or")
                    for i in range(len(screen_dict_by_nbr[main_block_nbr]['qualtrics_ids']))}
        else:
            screen_branch_logic = {i: dict(LogicType="Question", QuestionID=qualtrics_question_ids[i],
                                           QuestionIsInLoop="no",
                                           ChoiceLocator="q://{}/ChoiceNumericEntryValue/Rank/1"
                                           .format(qualtrics_question_ids[i]), Operator="EqualTo",
                                           QuestionIDFromLocator="{}".format(qualtrics_question_ids[i]),
                                           LeftOperand="q://{}/ChoiceNumericEntryValue/Rank/1"
                                           .format(qualtrics_question_ids[i]),
                                           RightOperand='2' if int(q_ids[i].split('--')[1]) == 0 else '1',
                                           Type="Expression",
                                           Description="Test for wrong answer in {}".format(q_ids[i]),
                                           Conjunction="Or")
                                   for i in range(len(qualtrics_question_ids))}
            screen_branch_logic["Type"] = "If"
            return screen_branch_logic

    def create_quad_survey(self, survey_name: str = 'API-Quadruple-Pilot', limit_q=None, q_to_display=14,
                           nbr_screen_qs=2, df_filename=QUAD_QS_FILENAME):
        self.create_triple_quad_survey(survey_name, limit_q, triple=False, q_to_display=q_to_display,
                                       nbr_screen_qs=nbr_screen_qs, df_filename=df_filename)

    def _create_triple_q_from_df_row(self, q, block_id, survey_id, dummy=False):
        """
        upload triple format question to the given survey_id
        :param q:
        :param block_id:
        :param survey_id:
        :return:
        """
        q_text = qualtrics_constants.triple_q_format.format(q[set_for_global.ANCHOR1_COL])
        choices = [q[set_for_global.ALTERNATIVE11_COL], q[set_for_global.ALTERNATIVE12_COL]]
        q_tag = q[set_for_global.ID_COL].replace('QQ', 'QT')
        if dummy:
            q_tag = "D" + q_tag
        q_params = self._build_question_params(q_text, q_id=q_tag, q_type=MULTIPLE_CHOICE_TYPE,
                                               answer_selector=MULTIPLE_CHOICE_SELECTOR, label=q_tag,
                                               forced_response=True,
                                               choices=choices, answer_sub_selector=MULTIPLE_CHOICE_SUBSELECTOR)
        _, qualtrics_id = self._create_question(survey_id, self._build_question_payload(q_params, survey_id), block_id)
        return qualtrics_id, q_tag

    def _create_quad_q_from_df_row(self, q, block_id, survey_id, dummy=False):
        q_text = qualtrics_constants.quad_q_format.format(q[set_for_global.ANCHOR1_COL],
                                                          q[set_for_global.ANCHOR2_COL])
        items = [q[set_for_global.ALTERNATIVE11_COL], q[set_for_global.ALTERNATIVE12_COL]]
        q_tag = q[set_for_global.ID_COL]
        if dummy:
            q_tag = "D" + q_tag
        q_params = self._build_question_params(q_text, q_id=q_tag, q_type=DRAG_RANK_TYPE,
                                               answer_selector=DRAG_RANK_SELECTOR, label=q_tag,
                                               forced_response=True,
                                               choices=items, answer_sub_selector=DRAG_RANK_SUBSELECTOR,
                                               groups="Ranking")
        _, qualtrics_id = self._create_question(survey_id, self._build_question_payload(q_params, survey_id), block_id)
        return qualtrics_id, q_tag

    def add_embed_data(self, survey_id, embed_name, embed_value=None, embed_type=None):
        self._validate_survey_id(survey_id)
        url = '{0}/{1}/{2}/{3}'.format(
            self.base_api_url,
            self.QUALTRICS_API_PATH_SURVEYS,
            survey_id,
            self.QUALTRICS_API_PATH_EMBEDDEDDATAFIELDS
        )

        body = {
            "name": embed_name,
            "embeddedDataFields": [
                {
                    "key": embed_name,
                }
            ]
        }

        if embed_value:
            body["embeddedDataFields"][0]["value"] = embed_value
        if embed_type:
            body["embeddedDataFields"][0]["type"] = embed_type

        response = requests.post(
            url,
            data=json.dumps(body),
            headers=self._build_headers('POST')
        )
        # logging.info(response.json())
        response.raise_for_status()

        return response.json()

    def add_prolific_id_embed(self, survey_id: str):
        self.add_embed_data(survey_id, "PROLIFIC_PID")

    def add_response_type_embed(self, survey_id: str):
        self.add_embed_data(survey_id, "Q_Response_Type", embed_value="In Process", embed_type='multiValueTextSet')

    @staticmethod
    def _build_question_params(html_q_text="", q_id="Q1", answer_selector=TEXT_ENTRY_SELECTOR, label="Test Question",
                               forced_response=False, q_type=TEXT_ENTRY_TYPE, choices=None, answer_sub_selector="",
                               items=None, groups=None, default_choice=None):
        question_params = {
            "text": html_q_text,
            "tag_number": q_id,
            "type": q_type,
            "translations": [],
            "answer_selector": answer_selector,
            "label": label,
            'answer_sub_selector': answer_sub_selector,
            'is_mandatory': forced_response,
            'block_number': 0,
        }

        # if choices is None:
        #     return question_params
        # else:
        if choices:
            question_params['choices'] = {
                "1": {
                    "Display": "{}".format(choices[0])
                },
                "2": {
                    "Display": "{}".format(choices[1])
                }
            }
            question_params["choice_order"] = ["1", "2"]

        if groups:
            question_params['groups'] = {
                "0": groups
                #     {
                #     "description": groups
                # }
            }

        if default_choice:
            question_params['default_choices'] = {
                "TEXT": {
                    "Text": "${e://Field/PROLIFIC_PID}"
                }
            }

        return question_params
