from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


class Workclass(str, Enum):
    StG = "State-gov"
    SEMI = "Self-emp-not-inc"
    Priv = "Private"
    FG = "Federal-gov"
    LG = "Local-gov"
    UNK = "?"
    SEI = "Self-emp-inc"
    WP = "Without-pay"
    NW = "Never-worked"


class Education(str, Enum):
    Bach = "Bachelors"
    HG = "HS-grad"
    Ele = "11th"
    Mast = "Masters"
    Nine = "9th"
    SC = "Some-college"
    AA = "Assoc-acdm"
    AV = "Assoc-voc"
    S_to_E = "7th-8th"
    Doc = "Doctorate"
    ProfS = "Prof-school"
    F_to_S = "5th-6th"
    Ten = "10th"
    F_to_F = "1st-4th"
    Pre = "Preschool"
    Twe = "12th"
    

class MaritalStatus(str, Enum):
    NM = "Never-married"
    MCS = "Married-civ-spouse"
    D = "Divorced"
    MSA = "Married-spouse-absent"
    S = "Separated"
    MAS = "Married-AF-spouse"
    W = "Widowed"


class Occupation(str, Enum):
    AC = "Adm-clerical"
    EM = "Exec-managerial"
    HC = "Handlers-cleaners"
    PS = "Prof-specialty"
    OS = "Other-service"
    S = "Sales"
    CR = "Craft-repair"
    TM = "Transport-moving"
    FF = "Farming-fishing"
    MOS = "Machine-op-inspct"
    TS = "Tech-support"
    Unk = "?"
    PrS = "Protective-serv"
    AF = "Armed-Forces"
    PHS = "Priv-house-serv"


class Relationship(str, Enum):
    NIF = "Not-in-family"
    H = "Husband"
    W = "Wife"
    OC = "Own-child"
    U = "Unmarried"
    OR = "Other-relative"


class Race(str, Enum):
    W = "White"
    B = "Black"
    API = "Asian-Pac-Islander"
    AIE = "Amer-Indian-Eskimo"
    O = "Other"


class Sex(str, Enum):
    M = "Male"
    F = "Female"


class NativeCountry(str, Enum):
    United_States = "United-States"
    Cuba = "Cuba"
    Jamaica = "Jamaica"
    India = "India"
    Unk = "?"
    Mexico = "Mexico"
    South = "South"
    Puerto_Rico = "Puerto-Rico"
    Honduras = "Honduras"
    England = "England"
    Canada = "Canada"
    Germany = "Germany"
    Iran = "Iran"
    Philippines = "Philippines"
    Italy = "Italy"
    Poland = "Poland"
    Columbia = "Columbia"
    Cambodia = "Cambodia"
    Thailand = "Thailand"
    Ecuador = "Ecuador"
    Laos = "Laos"
    Taiwan = "Taiwan"
    Haiti = "Haiti"
    Portugal = "Portugal"
    Dominican_Republic = "Dominican-Republic"
    El_Salvador = "El-Salvador"
    France = "France"
    Guatemala = "Guatemala"
    China = "China"
    Japan = "Japan"
    Yugoslavia = "Yugoslavia"
    Peru = "Peru"
    Outlying_US = "Outlying-US(Guam-USVI-etc)"
    Scotland = "Scotland"
    Trinadad_Tobago = "Trinadad&Tobago"
    Greece = "Greece"
    Nicaragua = "Nicaragua"
    Vietnam = "Vietnam"
    Hong = "Hong"
    Ireland = "Ireland"
    Hungary = "Hungary"
    Holand_Netherlands = "Holand-Netherlands"


class PersonData(BaseModel):
    age: int
    workclass: Workclass
    fnlgt: int
    education: Education
    education_num: int = Field(alias="education-num")
    marital_status: MaritalStatus = Field(alias="marital-status")
    occupation: Occupation
    relationship: Relationship
    race: Race
    sex: Sex
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: NativeCountry = Field(alias="native-country")

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
    })


